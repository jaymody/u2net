import torch


def set_weights(module: torch.nn.Module, random_state: torch.Tensor) -> None:
    torch.random.set_rng_state(random_state)
    for param in module.parameters():
        param.data = torch.randn(size=param.shape)


def same_parameters(net1: torch.nn.Module, net2: torch.nn.Module) -> bool:
    assert len(list(net1.parameters())) == len(list(net2.parameters()))
    for net1_param, net2_param in zip(net1.parameters(), net2.parameters()):
        if not torch.allclose(net1_param, net2_param):
            return False
    return True


def same_forward_pass(
    net1: torch.nn.Module, net2: torch.nn.Module, example_input: torch.Tensor
):
    random_state = torch.random.get_rng_state()
    set_weights(net1, random_state)
    set_weights(net2, random_state)
    assert same_parameters(net1, net2)
    return torch.allclose(net1(example_input), net2(example_input))
