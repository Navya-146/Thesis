from typing import List, Tuple

import torch.nn.init as init
from torch import nn

from main_code.FGR.src.datamodules.components.global_dicts import ACTIVATION_FUNCTIONS


def make_encoder_decoder(
    input_dim: int,
    hidden_dims: List[int],
    bottleneck_dim: int,
    activation: str,
) -> Tuple[nn.modules.container.Sequential, nn.modules.container.Sequential]:
    """Function for creating encoder and decoder models.

    Args:
        input_dim (int): Input dimension for encoder
        hidden_dims (List[int]): Dimensions for each layer
        bottleneck_dim (int): Dimension of bottleneck layer
        activation (str): Activation function to use

    Raises:
        ValueError: Activation function not supported

    Returns:
        Tuple[nn.modules.container.Sequential, nn.modules.container.Sequential]: Encoder and decoder models
    """  # noqa: E501

    encoder_layers = nn.ModuleList()
    decoder_layers = nn.ModuleList()
    output_dim = input_dim
    dec_shape = bottleneck_dim

    try:
        act_fn = ACTIVATION_FUNCTIONS[activation]()
    except KeyError:
        raise ValueError("Activation function not supported")

    for enc_dim in hidden_dims:
        encoder_layers.extend(
            [nn.Linear(input_dim, enc_dim), nn.LayerNorm(enc_dim), act_fn]
        )
        input_dim = enc_dim

    encoder_layers.append(nn.Linear(input_dim, bottleneck_dim))

    dec_dims = list(reversed(hidden_dims))
    for dec_dim in dec_dims:
        decoder_layers.extend(
            [nn.Linear(dec_shape, dec_dim), nn.LayerNorm(dec_dim), act_fn]
        )
        dec_shape = dec_dim

    decoder_layers.append(nn.Linear(dec_shape, output_dim))

    return nn.Sequential(*encoder_layers), nn.Sequential(*decoder_layers)


def make_predictor(
    fcn_input_dim: int,
    output_dims: List[int],
    activation: str,
    num_tasks: int,
    dropout: float,
) -> nn.modules.container.Sequential:
    """Function for creating predictor model.

    Args:
        fcn_input_dim (int): Input dimension for predictor
        output_dims (List[int]): Dimensions for each layer
        activation (str): Activation function to use
        num_tasks (int): Number of tasks for each dataset
        dropout (float): Dropout for each layer

    Raises:
        ValueError: Activation function not supported

    Returns:
        nn.modules.container.Sequential: Predictor model
    """

    try:
        act_fn = ACTIVATION_FUNCTIONS[activation]()
    except KeyError:
        raise ValueError("Activation function not supported")

    dropout_layer = nn.Dropout(dropout)

    layers = nn.ModuleList()
    for output_dim in output_dims:
        layers.extend(
            [
                nn.Linear(fcn_input_dim, output_dim),
                nn.LayerNorm(output_dim),
                act_fn,
            ]
        )
        fcn_input_dim = output_dim

    layers.extend([dropout_layer, nn.Linear(fcn_input_dim, num_tasks)])

    return nn.Sequential(*layers)


def tie_decoder_weights(encoder: nn.Sequential, decoder: nn.Sequential) -> None:
    """Function for tying weights of encoder and decoder.

    Args:
        encoder (nn.Sequential): Encoder model
        decoder (nn.Sequential): Decoder model
    """

    for i, encoder_layer in enumerate(reversed(encoder)):
        if isinstance(decoder[i], nn.Linear):
            decoder[i].weight = nn.Parameter(encoder_layer.weight.T)


def weight_init(model: nn.Module, activation: str) -> None:
    """Function for initializing weights.

    Args:
        model (nn.Module): Model to initialize
        activation (str): Activation function to use
    """
    if activation == "selu":
        nonlinearity = "linear"
    else:
        nonlinearity = "leaky_relu"

    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=nonlinearity)
            if m.bias is not None:
                init.constant_(m.bias, 0)
