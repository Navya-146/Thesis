from typing import List, Tuple

import finetuning_scheduler as fts
import torch

# from lightning.pytorch.callbacks import Callback
import torchmetrics
from lightning.pytorch import LightningModule
from torch import nn

from main_code.network import MultiViewNet, Net

# import rootutils
# rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class PhenoPredict(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        loss_function: torch.nn.Module,
        main_metric: torchmetrics.Metric,
        additional_metrics: torchmetrics.MetricCollection,
        compile: bool,
        task: str,
    ):
        super().__init__()

        self.save_hyperparameters(
            logger=False,
            ignore=["net", "loss_function", "main_metric", "additional_metrics"],
        )

        self.net = net
        self.criterion = loss_function

        # Metric objects
        self.train_metric = main_metric.clone()
        self.train_additional_metrics = additional_metrics.clone()
        self.train_additional_metrics.prefix = "train_"

        self.val_metric = main_metric.clone()
        self.val_additional_metrics = additional_metrics.clone()
        self.val_additional_metrics.prefix = "val_"

        self.test_metric = main_metric.clone()
        self.test_additional_metrics = additional_metrics.clone()
        self.test_additional_metrics.prefix = "test_"

    def forward(self, x):
        return self.net(x)

    @property
    def finetuningscheduler_callback(self) -> fts.FinetuningScheduler:
        fts_callback = [
            c for c in self.trainer.callbacks if isinstance(c, fts.FinetuningScheduler)
        ]
        return fts_callback[0] if fts_callback else None

    def on_train_start(self):
        self.val_metric.reset()
        self.val_additional_metrics.reset()

    def model_step(self, batch: Tuple[Tuple, torch.Tensor]):
        """A single step of the model.

        :param Tuple[Tuple, torch.Tensor] batch: _description_
        :return _type_: _description_
        """
        X, y = batch
        y_pred = self.forward(X)
        loss = self.criterion(y_pred, y)
        return loss, y_pred, y

    def training_step(self, batch, batch_idx):
        """Training step.

        :param _type_ batch: Batch
        :param _type_ batch_idx: _description_
        :return _type_: _description_
        """

        loss, y_pred, y = self.model_step(batch)

        self.train_loss = loss
        self.train_metric.update(y_pred, y)
        self.train_additional_metrics.update(y_pred, y)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_metric",
            self.train_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log_dict(
            self.train_additional_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def on_train_epoch_end(self) -> None:
        if self.finetuningscheduler_callback:
            self.log(
                "finetuning_schedule_depth",
                float(self.finetuningscheduler_callback.curr_depth),
            )

    def validation_step(self, batch, batch_idx) -> None:
        """Valdiation step.

        Parameters
        ----------
        batch : _type_
            _description_
        batch_idx : _type_
            index of batch
        """
        X, y = batch
        y_pred = self.forward(X)
        loss = self.criterion(y_pred, y)

        self.val_loss = loss
        self.val_metric.update(y_pred, y)
        self.val_additional_metrics.update(y_pred, y)

        # self.val_metric.compute()

        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_metric",
            self.val_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log_dict(
            self.val_additional_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def on_validation_epoch_end(self) -> None:
        self.val_metric.compute()
        self.val_additional_metrics.compute()

    def test_step(self, batch, batch_idx) -> None:
        """Test step."""

        X, y = batch
        y_pred = self.forward(X)
        loss = self.criterion(y_pred, y)

        self.test_loss = loss
        self.test_metric.update(y_pred, y)
        self.test_additional_metrics.update(y_pred, y)

        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "test_metric",
            self.test_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log_dict(
            self.test_additional_metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        Parameters
        ----------
        stage : str
             Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams["compile"] and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self):
        """Configuring the optimizers.

        Returns
        -------
        _type_
            _description_
        """

        optimizer = self.hparams["optimizer"](self.parameters())

        scheduler = self.hparams["scheduler"](optimizer)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class Netlightning(LightningModule):
    """A LightningModule subclass for regression tasks.

    Args:
        input_size (int): Size of the input layer.
        output_size (int): Size of the output layer.
        hidden_layers (List[int]): Sizes of the hidden layers.
        dropout (float): Dropout rate.
        lr (float): Learning rate.
        weight_decay (float): Weight decay.
        max_lr (float): Maximal learning rate.
        activation (str): Activation function.
        loss_function (str): Loss function. Must be one of ["mse", "mae", "huber"].

    Attributes:
        net (Net): The neural network architecture.
        train_r2 (R2Score): R2 score metric for training.
        val_r2 (R2Score): R2 score metric for validation.
        test_r2 (R2Score): R2 score metric for testing.
        train_mse (MeanSquaredError): Mean squared error metric for training.
        val_mse (MeanSquaredError): Mean squared error metric for validation.
        test_mse (MeanSquaredError): Mean squared error metric for testing.
        train_exp_var (ExplainedVariance): Explained variance metric for training.
        val_exp_var (ExplainedVariance): Explained variance metric for validation.
        test_exp_var (ExplainedVariance): Explained variance metric for testing.
        criterion (nn.Module): The loss function.
        lr (float): Learning rate.
        weight_decay (float): Weight decay.
        max_lr (float): Maximal learning rate.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: List[int],
        dropout: float,
        lr: float,
        weight_decay: float,
        max_lr: float,
        activation: str,
        loss_function: str,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.net = Net(
            hidden_layers=hidden_layers,
            input_size=input_size,
            output_size=output_size,
            dropout=dropout,
            activation=activation,
        )

        self.train_r2 = torchmetrics.R2Score()
        self.val_r2 = torchmetrics.R2Score()
        self.test_r2 = torchmetrics.R2Score()

        self.train_mse = torchmetrics.MeanSquaredError()
        self.val_mse = torchmetrics.MeanSquaredError()
        self.test_mse = torchmetrics.MeanSquaredError()

        self.train_exp_var = torchmetrics.ExplainedVariance()
        self.val_exp_var = torchmetrics.ExplainedVariance()
        self.test_exp_var = torchmetrics.ExplainedVariance()

        match loss_function:
            case "mse":
                self.criterion = nn.MSELoss()
            case "mae":
                self.criterion = nn.L1Loss()
            case "huber":
                self.criterion = nn.HuberLoss()
            case _:
                self.criterion = nn.MSELoss()

        # Learning parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_lr = max_lr

    def forward(self, x):
        """Forward pass through the neural network.

        Args:
            x: The input tensor.

        Returns:
            The output tensor of the neural network.
        """
        return self.net(x)

    def configure_optimizers(self):
        """Returns the configured optimizers and schedulers for training the model.

        Returns:
            Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]: A tuple containing the configured optimizers and schedulers.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        """Executes a single training step for the model.

        Args:
            batch: A tuple containing the input data (X) and the corresponding labels (y).
            batch_idx: An integer representing the current batch index.

        Returns:
            The calculated loss value.
        """
        X, y = batch
        y_pred = self.forward(X)
        loss = self.criterion(y_pred, y)

        self.train_mse.update(y_pred, y)
        self.train_r2.update(y_pred, y)
        self.train_exp_var.update(y_pred, y)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_mse",
            self.train_mse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_r2",
            self.train_r2,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "train_exp_var",
            self.train_exp_var,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """Executes a single validation step in the training loop.

        Args:
            batch (tuple): A tuple containing the input features and the corresponding labels.
            batch_idx (int): The index of the current batch.

        Returns:
            None
        """
        X, y = batch
        y_pred = self.forward(X)
        loss = self.criterion(y_pred, y)

        self.val_mse.update(y_pred, y)
        self.val_r2.update(y_pred, y)
        self.val_exp_var.update(y_pred, y)

        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        self.log(
            "val_mse",
            self.val_mse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "val_r2",
            self.val_r2,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "val_exp_var",
            self.val_exp_var,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return None

    def test_step(self, batch, batch_idx):
        """Executes a test step in the training loop.

        Args:
            batch: The input batch containing the features and labels.
            batch_idx: The index of the current batch.

        Returns:
            None
        """
        X, y = batch
        y_pred = self.forward(X)
        loss = self.criterion(y_pred, y)

        self.test_mse.update(y_pred, y)
        self.test_r2.update(y_pred, y)
        self.test_exp_var.update(y_pred, y)

        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        self.log(
            "test_mse",
            self.test_mse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "test_r2",
            self.test_r2,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "test_exp_var",
            self.test_exp_var,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return None


class NetMultiViewLightning(LightningModule):
    """This class implements a neural network model for multi-view data using PyTorch Lightning.

    Args:
        layers_before_concat (List[int]): A list of integers representing the number of units in each layer before concatenation.
        layers_after_concat (List[int]): A list of integers representing the number of units in each layer after concatenation.
        input_size (List[int]): A list of integers representing the size of the input for each view.
        output_size (int): The size of the output.
        dropout (float): The dropout rate.
        activation (str): The activation function to use.
        lr (float): The learning rate for the optimizer.
        max_lr (float): The maximum learning rate for the one-cycle learning rate scheduler.
        weight_decay (float): The weight decay for the optimizer.
        task (str, optional): The task of the model. Defaults to "regression".
        loss_function (str, optional): The loss function to use. Defaults to "mse".
    """

    def __init__(
        self,
        layers_before_concat: List[int],
        layers_after_concat: List[int],
        input_size: List[int],
        output_size: int,
        dropout: float,
        activation: str,
        lr: float,
        max_lr: float,
        weight_decay: float,
        task: str = "regression",
        loss_function: str = "mse",
    ):
        super().__init__()

        self.save_hyperparameters()

        if task == "regression":
            # Map loss function names to their respective loss functions
            loss_functions = {
                "mse": nn.MSELoss(),
                "mae": nn.L1Loss(),
                "huber": nn.HuberLoss(),
            }

            # Check if the loss function is valid
            if loss_function not in loss_functions:
                raise ValueError("Loss function must be one of ['mse', 'mae', 'huber']")

            # Set the criterion based on the loss function
            self.criterion = loss_functions[loss_function]

            self.train_mae = torchmetrics.MeanAbsoluteError()
            self.val_mae = torchmetrics.MeanAbsoluteError()
            self.test_mae = torchmetrics.MeanAbsoluteError()

            self.train_r2 = torchmetrics.R2Score()
            self.val_r2 = torchmetrics.R2Score()
            self.test_r2 = torchmetrics.R2Score()

            self.train_corr = torchmetrics.SpearmanCorrCoef()
            self.val_corr = torchmetrics.SpearmanCorrCoef()
            self.test_corr = torchmetrics.SpearmanCorrCoef()

        elif task == "binary_classification":
            loss_functions = {
                "bce": nn.BCEWithLogitsLoss(),
                "cross_entropy": nn.CrossEntropyLoss(),
            }

            self.train_acc = torchmetrics.Accuracy(task="binary")
            self.val_acc = torchmetrics.Accuracy(task="binary")
            self.test_acc = torchmetrics.Accuracy(task="binary")

            self.train_auc_roc = torchmetrics.AUROC(task="binary")
            self.val_auc_roc = torchmetrics.AUROC(task="binary")
            self.test_auc_roc = torchmetrics.AUROC(task="binary")

            self.train_f1 = torchmetrics.F1Score(task="binary")
            self.val_f1 = torchmetrics.F1Score(task="binary")
            self.test_f1 = torchmetrics.F1Score(task="binary")

            if loss_function not in loss_functions:
                raise ValueError(
                    "Loss function must be one of ['bce', 'cross_entropy']"
                )

            self.criterion = loss_functions[loss_function]

        else:
            raise ValueError(
                "Task must be one of ['regression', 'binary_classification']"
            )

        self.task = task

        # Defining model
        self.net = MultiViewNet(
            layers_before_concat=layers_before_concat,
            layers_after_concat=layers_after_concat,
            input_size=input_size,
            output_size=output_size,
            dropout=dropout,
            activation=activation,
        )

        # Learning Parameters
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_lr = max_lr

    def forward(self, x):
        """Calculates the forward pass of the neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.net(x)

    def configure_optimizers(self):
        """Configures the optimizers for the model.

        Returns:
            A tuple containing two lists:
                - The first list contains the optimizer(s) for the model.
                - The second list contains the scheduler(s) for the optimizer(s).
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        """Executes a training step for the model.

        Args:
            batch (tuple): A tuple containing the input features and corresponding labels.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss value.
        """
        X, y = batch
        y_pred = self.forward(X)
        loss = self.criterion(y_pred, y)

        if self.task == "regression":
            self.train_r2.update(y_pred, y)
            self.train_mae.update(y_pred, y)
            self.train_corr.update(y_pred, y)

            self.log("train_r2", self.train_r2, on_epoch=True, prog_bar=True)
            self.log("train_mae", self.train_mae, on_epoch=True, prog_bar=True)
            self.log("train_corr", self.train_corr, on_epoch=True, prog_bar=True)

        elif self.task == "binary_classification":
            self.train_acc.update(y_pred, y)
            self.train_auc_roc.update(y_pred, y)
            self.train_f1.update(y_pred, y)

            self.log("train_acc", self.train_acc, on_epoch=True, prog_bar=True)
            self.log("train_auc_roc", self.train_auc_roc, on_epoch=True, prog_bar=True)
            self.log("train_f1", self.train_f1, on_epoch=True, prog_bar=True)

        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Performs a validation step in the training loop.

        Args:
            batch: A tuple containing the input data (X) and the target labels (y).
            batch_idx: An integer representing the index of the current batch.

        Returns:
            None
        """
        X, y = batch
        y_pred = self.forward(X)
        loss = self.criterion(y_pred, y)

        if self.task == "regression":
            self.val_r2.update(y_pred, y)
            self.val_mae.update(y_pred, y)
            self.val_corr.update(y_pred, y)
            self.log("val_r2", self.val_r2, on_epoch=True)
            self.log("val_mae", self.val_mae, on_epoch=True)
            self.log("val_corr", self.val_corr, on_epoch=True)

        elif self.task == "binary_classification":
            self.val_acc.update(y_pred, y)
            self.val_auc_roc.update(y_pred, y)
            self.val_f1.update(y_pred, y)
            self.log("val_acc", self.val_acc, on_epoch=True)
            self.log("val_auc_roc", self.val_auc_roc, on_epoch=True)
            self.log("val_f1", self.val_f1, on_epoch=True)

        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

    def test_step(self, batch, batch_idx):
        """Perform a test step in the training loop.

        Args:
            batch (tuple): A tuple containing the input data and labels.
            batch_idx (int): The index of the current batch.

        Returns:
            None

        Raises:
            None
        """
        X, y = batch
        y_pred = self.forward(X)
        loss = self.criterion(y_pred, y)

        if self.task == "regression":
            self.test_r2.update(y_pred, y)
            self.test_mae.update(y_pred, y)
            self.test_corr.update(y_pred, y)

            self.log("test_r2", self.test_r2, on_epoch=True, prog_bar=True)
            self.log("test_mae", self.test_mae, on_epoch=True, prog_bar=True)
            self.log("test_corr", self.test_corr, on_epoch=True, prog_bar=True)

        elif self.task == "binary_classification":
            self.test_acc.update(y_pred, y)
            self.test_auc_roc.update(y_pred, y)
            self.test_f1.update(y_pred, y)

            self.log("test_acc", self.test_acc, on_epoch=True, prog_bar=True)
            self.log("test_auc_roc", self.test_auc_roc, on_epoch=True, prog_bar=True)
            self.log("test_f1", self.test_f1, on_epoch=True, prog_bar=True)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True)


if __name__ == "__main__":
    # from data import EncodeModule

    # torch.set_float32_matmul_precision("high")

    # # model = Netlightning(
    # #     input_size=6078,
    # #     output_size=1,
    # #     hidden_layers=[100, 100],
    # #     dropout=0.5,
    # #     lr=1e-3,
    # #     weight_decay=1e-5,
    # #     max_lr=1e-2,
    # #     activation="relu",
    # #     loss_function="mse",
    # # )

    # # Small test

    # model = NetMultiViewLightning(
    #     layers_before_concat=[100, 100],
    #     layers_after_concat=[100, 100],
    #     input_size=[6014, 64],
    #     output_size=1,
    #     dropout=0.5,
    #     activation="relu",
    #     lr=1e-3,
    #     max_lr=1e-2,
    #     weight_decay=1e-5,
    #     task="binary_classification",
    #     loss_function="bce",
    # )

    # # datamodule = EncodeModule(
    # #     path = "/storage/bt20d204/data/regression_data/bloom2013_regression.feather"
    # # )

    # datamodule = EncodeModule(
    #     path="/storage/bt20d204/data/bloom2013_clf_3_pubchem.feather", num_workers=1
    # )

    # trainer = Trainer(accelerator="gpu", devices=1, fast_dev_run=100)

    # trainer.fit(model, datamodule)
    pass
