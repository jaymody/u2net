import torch

import u2net
import u2net_original
from utils import same_forward_pass, same_parameters, set_weights


def test_upsample_like():
    src = torch.randn(size=(4, 32, 16, 16))
    tar = torch.randn(size=(4, 32, 32, 32))
    assert torch.allclose(
        u2net._upsample_like(src, tar),
        u2net_original._upsample_like(src, tar),
    )


def test_ConvLayer():
    example_input = torch.randn(size=(4, 3, 512, 512))
    assert same_forward_pass(
        u2net.ConvLayer(3, 1, dirate=1),
        u2net_original.REBNCONV(3, 1, dirate=1),
        example_input,
    )
    assert same_forward_pass(
        u2net.ConvLayer(3, 3, dirate=1),
        u2net_original.REBNCONV(3, 3, dirate=1),
        example_input,
    )
    assert same_forward_pass(
        u2net.ConvLayer(3, 1, dirate=2),
        u2net_original.REBNCONV(3, 1, dirate=2),
        example_input,
    )


def test_RSU():
    example_input = torch.randn(size=(4, 3, 512, 512))
    assert same_forward_pass(
        u2net_original.RSU7(3, 16, 1), u2net.RSU(3, 16, 1, height=7), example_input
    )
    assert same_forward_pass(
        u2net_original.RSU6(3, 16, 1), u2net.RSU(3, 16, 1, height=6), example_input
    )
    assert same_forward_pass(
        u2net_original.RSU5(3, 16, 1), u2net.RSU(3, 16, 1, height=5), example_input
    )
    assert same_forward_pass(
        u2net_original.RSU4(3, 16, 1), u2net.RSU(3, 16, 1, height=4), example_input
    )


def test_RSUF():
    example_input = torch.randn(size=(4, 3, 512, 512))
    assert same_forward_pass(
        u2net_original.RSU4F(3, 16, 1), u2net.RSUF(3, 16, 1, height=4), example_input
    )


def test_U2Net():
    example_input = torch.randn(size=(4, 3, 512, 512))
    net1, net2 = u2net_original.U2NET(3, 1), u2net.U2Net(3, 1, large=True)

    # can't use same_forward_pass since output of U2Net is a list of tensors and not
    # a single tensor
    random_state = torch.random.get_rng_state()
    set_weights(net1, random_state)
    set_weights(net2, random_state)
    assert same_parameters(net1, net2)

    out1 = net1(example_input)
    out2 = net2(example_input)
    assert len(out1) == len(out2)
    for tensor1, tensor2 in zip(out1, out2):
        assert torch.allclose(tensor1, tensor2)


def test_U2Net_small():
    example_input = torch.randn(size=(4, 3, 512, 512))
    net1, net2 = u2net_original.U2NETP(3, 1), u2net.U2Net(3, 1, large=False)

    # can't use same_forward_pass since output of U2Net is a list of tensors and not
    # a single tensor
    random_state = torch.random.get_rng_state()
    set_weights(net1, random_state)
    set_weights(net2, random_state)
    assert same_parameters(net1, net2)

    out1 = net1(example_input)
    out2 = net2(example_input)
    assert len(out1) == len(out2)
    for tensor1, tensor2 in zip(out1, out2):
        assert torch.allclose(tensor1, tensor2)
