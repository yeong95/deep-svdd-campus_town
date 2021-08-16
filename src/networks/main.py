from .resnet import Encoder, Autoencoder
from .efficientnet_autoencoder import Efficientnet_encoder, Efficientnet_autoencoder


def build_network(net_name):
    """Builds the neural network."""

    implemented_networks = ('resnet', 'efficientnet')
    assert net_name in implemented_networks

    net = None

    if net_name == 'resnet':
        net = Encoder()

    if net_name == 'efficientnet':
        net = Efficientnet_encoder()

    return net


def build_autoencoder(net_name):
    """Builds the corresponding autoencoder network."""

    implemented_networks = ('resnet', 'efficientnet')
    assert net_name in implemented_networks

    ae_net = None

    
    if net_name == 'resnet':
        ae_net = Autoencoder()

    if net_name == 'efficientnet':
        ae_net = Efficientnet_autoencoder()

    return ae_net
