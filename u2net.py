import torch
import torch.nn as nn
import torch.nn.functional as F


def _upsample_like(src, tar):
    return F.interpolate(src, size=tar.shape[2:], mode="bilinear", align_corners=False)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dirate):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1 * dirate,
            dilation=1 * dirate,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class PoolLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

    def forward(self, x):
        out = self.layer(x)
        out_pooled = self.pool(out)
        return out, out_pooled


class RSU(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, height):
        super().__init__()
        assert height > 1

        n_layers = height - 2
        self.convin = ConvLayer(in_channels, out_channels, dirate=1)
        self.conv_encoders = nn.ModuleList(
            [PoolLayer(ConvLayer(out_channels, hidden_channels, dirate=1))]
            + [
                PoolLayer(ConvLayer(hidden_channels, hidden_channels, dirate=1))
                for _ in range(n_layers)
            ]
        )
        self.conv_last = ConvLayer(hidden_channels, hidden_channels, dirate=2)
        self.conv_decoders = nn.ModuleList(
            [
                ConvLayer(hidden_channels * 2, hidden_channels, dirate=1)
                for _ in range(n_layers)
            ]
            + [ConvLayer(hidden_channels * 2, out_channels, dirate=1)]
        )

    def forward(self, x):
        xin = self.convin(x)

        x_pooled = xin
        decoder_inputs = []
        for conv in self.conv_encoders:
            x, x_pooled = conv(x_pooled)
            decoder_inputs.append(x)

        assert len(self.conv_decoders) == len(decoder_inputs)
        for i, (conv, dec_x) in enumerate(
            list(zip(self.conv_decoders, reversed(decoder_inputs)))
        ):
            dec_in = self.conv_last(dec_x) if i == 0 else _upsample_like(dec_in, dec_x)
            dec_in = torch.cat((dec_in, dec_x), 1)
            dec_in = conv(dec_in)

        return dec_in + xin


class RSUF(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, height):
        super().__init__()
        assert height > 1

        # difference to RSU
        #   - no ConvLayer instead of ConvLayerPool (no pooling)
        #   - no _upsample_like
        #   - dirate grows and shrinks (example: 1, 2, 4, 8, 4, 2, 1)

        n_layers = height - 2
        self.convin = ConvLayer(in_channels, out_channels, dirate=1)
        self.conv_encoders = nn.ModuleList(
            [ConvLayer(out_channels, hidden_channels, dirate=1)]
            + [
                ConvLayer(hidden_channels, hidden_channels, dirate=2 ** (i + 1))
                for i in range(n_layers)
            ]
        )
        self.conv_last = ConvLayer(
            hidden_channels, hidden_channels, dirate=2 ** (n_layers + 1)
        )
        self.conv_decoders = nn.ModuleList(
            [
                ConvLayer(
                    hidden_channels * 2,
                    hidden_channels,
                    dirate=2 ** (n_layers - i),
                )
                for i in range(n_layers)
            ]
            + [ConvLayer(hidden_channels * 2, out_channels, dirate=1)]
        )

    def forward(self, x):
        xin = self.convin(x)

        x = xin
        decoder_inputs = []
        for conv in self.conv_encoders:
            x = conv(x)
            decoder_inputs.append(x)

        assert len(self.conv_decoders) == len(decoder_inputs)
        for i, (conv, dec_x) in enumerate(
            list(zip(self.conv_decoders, reversed(decoder_inputs)))
        ):
            dec_in = self.conv_last(dec_x) if i == 0 else dec_in
            dec_in = torch.cat((dec_in, dec_x), 1)
            dec_in = conv(dec_in)

        return dec_in + xin


class U2Net(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, large=True):
        super().__init__()

        in_ch, out_ch = in_channels, out_channels
        self.encoder_layers = nn.ModuleList(
            [
                PoolLayer(RSU(in_ch, 32, 64, 7) if large else RSU(in_ch, 16, 64, 7)),
                PoolLayer(RSU(64, 32, 128, 6) if large else RSU(64, 16, 64, 6)),
                PoolLayer(RSU(128, 64, 256, 5) if large else RSU(64, 16, 64, 5)),
                PoolLayer(RSU(256, 128, 512, 4) if large else RSU(64, 16, 64, 4)),
                PoolLayer(RSUF(512, 256, 512, 4) if large else RSUF(64, 16, 64, 4)),
            ]
        )
        self.middle_layer = RSUF(512, 256, 512, 4) if large else RSUF(64, 16, 64, 4)

        # decoder
        self.decoder_layers = nn.ModuleList(
            [
                RSUF(1024, 256, 512, 4) if large else RSUF(128, 16, 64, 4),
                RSU(1024, 128, 256, 4) if large else RSU(128, 16, 64, 4),
                RSU(512, 64, 128, 5) if large else RSU(128, 16, 64, 5),
                RSU(256, 32, 64, 6) if large else RSU(128, 16, 64, 6),
                RSU(128, 16, 64, 7),
            ]
        )

        # sides
        self.side_layers = nn.ModuleList(
            [
                nn.Conv2d(64, out_ch, 3, padding=1),
                nn.Conv2d(64, out_ch, 3, padding=1),
                nn.Conv2d(128 if large else 64, out_ch, 3, padding=1),
                nn.Conv2d(256 if large else 64, out_ch, 3, padding=1),
                nn.Conv2d(512 if large else 64, out_ch, 3, padding=1),
                nn.Conv2d(512 if large else 64, out_ch, 3, padding=1),
            ]
        )

        self.outconv = nn.Conv2d(len(self.side_layers) * out_ch, out_ch, 1)

    def forward(self, x):
        x_pooled = x
        decoder_inputs = []
        for layer in self.encoder_layers:
            x, x_pooled = layer(x_pooled)
            decoder_inputs.append(x)

        side_inputs = []
        assert len(self.decoder_layers) == len(decoder_inputs)
        for i, (layer, dec_x) in enumerate(
            list(zip(self.decoder_layers, reversed(decoder_inputs)))
        ):
            if i == 0:
                dec_in = self.middle_layer(x_pooled)
                side_inputs.append(dec_in)
            dec_in = _upsample_like(dec_in, dec_x)
            dec_in = torch.cat((dec_in, dec_x), 1)
            dec_in = layer(dec_in)
            side_inputs.append(dec_in)

        outputs = []
        assert len(self.side_layers) == len(side_inputs)
        for i, (layer, side_x) in enumerate(
            list(zip(self.side_layers, reversed(side_inputs)))
        ):
            d = layer(side_x)
            if i != 0:
                d = _upsample_like(d, outputs[0])
            outputs.append(d)

        outputs = [self.outconv(torch.cat([*outputs], 1))] + outputs
        outputs = [torch.sigmoid(out) for out in outputs]
        return outputs
