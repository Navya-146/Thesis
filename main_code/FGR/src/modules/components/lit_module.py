from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule


class BaseLitModule(LightningModule):
    def __init__(
        self,
        network: DictConfig,
        loss_metric: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """BaseLightningModule.

        Args:
            network (DictConfig): Network config.
            loss_metric (DictConfig): Loss and metric config.
            optimizer (DictConfig): Optimizer config.
            scheduler (DictConfig): Scheduler config.
            logging (DictConfig): Logging config.
            args (Any): Additional arguments for pytorch_lightning.LightningModule.
            kwargs (Any): Additional keyword arguments for pytorch_lightning.LightningModule.
        """  # noqa: E501

        super().__init__(*args, **kwargs)
        self.model = hydra.utils.instantiate(network.model)
        self.opt_params = optimizer
        self.slr_params = scheduler
        self.logging_params = loss_metric.logging

    def forward(self, x: Any) -> Any:
        return self.model.forward(*x)

    def ubc_loss(self, z_d: Any) -> Any:
        cov = torch.cov(z_d)
        off_diag = cov - torch.diag(torch.diag(cov))
        ubc_loss = torch.sum(off_diag**2)
        return ubc_loss

    def configure_optimizers(self) -> Any:
        base_optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
            self.opt_params.base, _partial_=True
        )
        optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
            self.opt_params.sam,
            params=self.parameters(),
            base_optimizer=base_optimizer,
            _convert_="partial",
        )
        if self.slr_params.get("scheduler"):
            scheduler: torch.optim.lr_scheduler._LRScheduler = hydra.utils.instantiate(
                self.slr_params.scheduler,
                optimizer=optimizer,  # type: ignore
                total_steps=self.trainer.estimated_stepping_batches,
                _convert_="partial",
            )
            lr_scheduler_dict = {"scheduler": scheduler}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
        return {"optimizer": optimizer}
