from typing import List

import torch
from torch import nn

from main_code.FGR.src.datamodules.components.global_dicts import (
    MFG_INPUT_DIM,
    TASK_DICT,
)
from main_code.FGR.src.modules.models.utils import (
    make_encoder_decoder,
    make_predictor,
    tie_decoder_weights,
    weight_init,
)


class FGRModel(nn.Module):
    def __init__(
        self,
        fg_input_dim: int,
        num_feat_dim: int,
        method: str,
        tokenize_dataset: str,
        frequency: int,
        dataset: str,
        descriptors: bool,
        hidden_dims: List[int],
        bottleneck_dim: int,
        output_dims: List[int],
        dropout: float,
        activation: str,
        tie_weights: bool,
    ) -> None:
        """Model for FGR.

        Args:
            fg_input_dim (int): Input dimension for FG
            num_feat_dim (int): Number of features for descriptors
            method (str): Representation method to train
            tokenize_dataset (str): Tokenization dataset
            frequency (int): Frequency for tokenization
            dataset (str): Dataset to train on
            descriptors (bool): Whether to use descriptors
            hidden_dims (List[int]): Dimensions for each layer
            bottleneck_dim (int): Dimension of bottleneck layer
            output_dims (List[int]): Dimensions for each predictor layer
            dropout (float): Dropout for each layer
            activation (str): Activation function to use
            tie_weights (bool): Whether to tie weights of encoder and decoder

        Raises:
            ValueError: Method not supported
        """
        super().__init__()

        self.method = method
        self.descriptors = descriptors
        self.tie_weights = tie_weights
        self.num_tasks, self.task_type, self.regression = TASK_DICT[dataset]

        if self.method == "FG":
            input_dim = fg_input_dim
        elif self.method == "MFG":
            input_dim = MFG_INPUT_DIM[tokenize_dataset][frequency]
        elif self.method == "FGR":
            input_dim = fg_input_dim + MFG_INPUT_DIM[tokenize_dataset][frequency]
        else:
            raise ValueError("Method not supported")
        self.encoder, self.decoder = make_encoder_decoder(
            input_dim, hidden_dims, bottleneck_dim, activation
        )

        # Tie the weights of the encoder and decoder
        if self.tie_weights:
            tie_decoder_weights(self.encoder, self.decoder)

        if self.descriptors:
            fcn_input_dim = bottleneck_dim + num_feat_dim
        else:
            fcn_input_dim = bottleneck_dim

        self.predictor = make_predictor(
            fcn_input_dim, output_dims, activation, self.num_tasks, dropout
        )

        # Initialize weights
        weight_init(self.encoder, activation)
        weight_init(self.decoder, activation)
        weight_init(self.predictor, activation)

    def forward(self, x, num_feat=None):
        """Perform forward pass."""
        z_d = self.encoder(x)
        x_hat = self.decoder(z_d)
        if self.descriptors:
            assert num_feat is not None, "Descriptors not provided"
            z_d = torch.cat([z_d, num_feat], dim=1)  # Concatenate descriptors
        output = self.predictor(z_d)
        return output, z_d, x_hat


class FGRPretrainModel(nn.Module):
    def __init__(
        self,
        fg_input_dim: int,
        method: str,
        tokenize_dataset: str,
        frequency: int,
        hidden_dims: List[int],
        bottleneck_dim: int,
        activation: str,
        tie_weights: bool,
    ) -> None:
        """Pretrain model for FGR.

        Args:
            fg_input_dim (int): Input dimension for FG
            method (str): Representation method to train
            tokenize_dataset (str): Tokenization dataset
            frequency (int): Frequency for tokenization
            hidden_dims (List[int]): Dimensions for each layer
            bottleneck_dim (int): Dimension of bottleneck layer
            activation (str): Activation function to use
            tie_weights (bool): Whether to tie weights of encoder and decoder

        Raises:
            ValueError: Method not supported
        """
        super().__init__()

        self.tie_weights = tie_weights

        if method == "FG":
            input_dim = fg_input_dim
        elif method == "MFG":
            input_dim = MFG_INPUT_DIM[tokenize_dataset][frequency]
        elif method == "FGR":
            input_dim = fg_input_dim + MFG_INPUT_DIM[tokenize_dataset][frequency]
        else:
            raise ValueError("Method not supported")
        self.encoder, self.decoder = make_encoder_decoder(
            input_dim, hidden_dims, bottleneck_dim, activation
        )
        # Tie the weights of the encoder and decoder
        if self.tie_weights:
            tie_decoder_weights(self.encoder, self.decoder)

    def forward(self, x):
        """Perform forward pass."""
        z_d = self.encoder(x)
        x_hat = self.decoder(z_d)
        return z_d, x_hat
