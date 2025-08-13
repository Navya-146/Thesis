import numpy as np
import seaborn as sns


def set_theme() -> None:
    """Consistency is Key."""
    sns.set_theme(
        context="notebook",
        style="white",
        palette="colorblind",
        rc={
            "figure.figsize": (8, 6),
            "xtick.labelsize": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 16,
            "ytick.labelsize": 12,
            "figure.dpi": 120,
        },
    )

    sns.set_palette(
        [
            "#000000",
            "#E69F00",
            "#56B4E9",
            "#009E73",
            "#F0E442",
            "#0072B2",
            "#D55E00",
            "#CC79A7",
        ]
    )
    np.set_printoptions(precision=4)


def predict(model, X, clf=False, device="cuda:0"):
    """Predicts the output of a given model on a given input tensor.

    Parameters:
        model (nn.Module): The model to use for prediction.
        X (torch.Tensor): The input tensor.
        clf (bool, optional): Whether to use the model as a classifier. If True, the predicted output will be the index of the highest value in each row of the output tensor. Defaults to False.
        device (str, optional): The device to use for computation. Defaults to "cuda:0".

    Returns:
        torch.Tensor: The predicted output tensor.
    """
    y_pred = model(X.to(device))
    # if clf:
    #     y_pred = torch.argmax(y_pred, dim=1)
    return y_pred
