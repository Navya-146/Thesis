from typing import Any

from omegaconf import DictConfig

from main_code.FGR.src.modules.components.lit_module import BaseLitModule
from main_code.FGR.src.modules.losses import load_loss
from main_code.FGR.src.modules.metrics import load_metrics


class FGRLitModule(BaseLitModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Model loop (model_step)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        network: DictConfig,
        loss_metrics: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """LightningModule with standalone train, val and test dataloaders.

        Args:
            network (DictConfig): Network config.
            loss_metric (DictConfig): Loss and metrics config.
            optimizer (DictConfig): Optimizer config.
            scheduler (DictConfig): Scheduler config.
            args (Any): Additional arguments for pytorch_lightning.LightningModule.
            kwargs (Any): Additional keyword arguments for pytorch_lightning.LightningModule.
        """

        super().__init__(network, loss_metrics, optimizer, scheduler, *args, **kwargs)
        self.recon_loss = load_loss(network.recon_loss)
        self.loss = load_loss(loss_metrics.loss)
        self.loss_weights = network.loss_weight

        main_metric, valid_metric_best, add_metrics = load_metrics(loss_metrics.metrics)
        self.train_metric = main_metric.clone()
        self.train_add_metrics = add_metrics.clone(postfix="/train")
        self.valid_metric = main_metric.clone()
        self.valid_metric_best = valid_metric_best.clone()
        self.valid_add_metrics = add_metrics.clone(postfix="/valid")
        self.test_metric = main_metric.clone()
        self.test_add_metrics = add_metrics.clone(postfix="/test")

        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False

    def model_step(self, batch: Any, *args: Any, **kwargs: Any) -> Any:
        targets = batch[-1]
        logits, z_d, x_hat = self.forward(batch[:-1])
        loss = self.loss(logits, targets)
        loss += self.loss_weights["recon_loss"] * self.recon_loss(x_hat, batch[0])
        loss += self.loss_weights["ubc_loss"] * self.ubc_loss(z_d)
        return loss, logits, targets

    def on_train_start(self) -> None:
        # by default lightning executes validation step sanity checks before
        # training starts, so we need to make sure valid_metric_best doesn't store
        # accuracy from these checks
        self.valid_metric_best.reset()

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        optimizer = self.optimizers()

        # first forward-backward pass
        loss_1, logits, targets = self.model_step(batch, batch_idx)
        self.manual_backward(loss_1)
        optimizer.first_step(zero_grad=True)  # clear gradients # type: ignore

        # second forward-backward pass
        loss_2 = self.model_step(batch, batch_idx)[0]
        self.manual_backward(loss_2)
        optimizer.second_step(zero_grad=True)  # clear gradients # type: ignore

        scheduler = self.lr_schedulers()
        scheduler.step()  # type: ignore

        self.log(
            f"{self.loss.__class__.__name__}/train",
            loss_1,
            **self.logging_params,
        )

        self.train_metric.update(logits, targets)
        self.log(
            f"{self.train_metric.__class__.__name__}/train",
            self.train_metric,
            **self.logging_params,
        )

        self.train_add_metrics.update(logits, targets)
        self.log_dict(self.train_add_metrics, **self.logging_params)  # type: ignore

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        loss, logits, targets = self.model_step(batch, batch_idx)
        self.log(
            f"{self.loss.__class__.__name__}/valid",
            loss,
            **self.logging_params,
        )

        self.valid_metric.update(logits, targets)
        self.log(
            f"{self.valid_metric.__class__.__name__}/valid",
            self.valid_metric,
            **self.logging_params,
        )

        self.valid_add_metrics.update(logits, targets)
        self.log_dict(self.valid_add_metrics, **self.logging_params)  # type: ignore
        return {"loss": loss, "preds": logits, "targets": targets}

    def on_validation_epoch_end(self) -> None:
        valid_metric = self.valid_metric.compute()  # get current valid metric
        self.valid_metric_best(valid_metric)  # update best so far valid metric
        # log `valid_metric_best` as a value through `.compute()` method, instead
        # of as a metric object otherwise metric would be reset by lightning
        # after each epoch
        self.log(
            f"{self.valid_metric.__class__.__name__}/valid_best",
            self.valid_metric_best.compute(),
            **self.logging_params,
        )

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        loss, logits, targets = self.model_step(batch, batch_idx)
        self.log(f"{self.loss.__class__.__name__}/test", loss, **self.logging_params)

        self.test_metric.update(logits, targets)
        self.log(
            f"{self.test_metric.__class__.__name__}/test",
            self.test_metric,
            **self.logging_params,
        )

        self.test_add_metrics.update(logits, targets)
        self.log_dict(self.test_add_metrics, **self.logging_params)  # type: ignore
        return {"loss": loss, "preds": logits, "targets": targets}

    def on_test_epoch_end(self) -> None:
        pass
