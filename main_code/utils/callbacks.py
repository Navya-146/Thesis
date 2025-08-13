# Writing a Custom Pruning Callback for KFold Training
# References:
#   https://github.com/optuna/optuna/blob/master/optuna/integration/pytorch_lightning.py
#   https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.Callback.html
#
# Going to keep it simple. Not suitable for Distributed Data Parallel (DDP) training!

import lightning.pytorch as pl
import optuna
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


class KFoldPruningCallback(Callback):
    """A Callback for pruning unpromising trials in a K-Fold training setting.

    Args:
        trial:
            A `optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning. Must be present in the `lightning.pytorch.LightningModule` tracked metrics.
        num_folds_before_pruning:
            Total number of folds in the K-Fold training before pruning the entire trial, e.g. if `num_folds_before_pruning=3`,
            the trial will be pruned if the score is not improved in the last 3 folds.

    .. note::
        Do not use it with DistributedDataParallel.
    """

    def __init__(
        self, trial: optuna.trial.Trial, monitor: str, num_folds_before_pruning: int = 3
    ) -> None:
        super().__init__()

        self._trial = trial
        self.monitor = monitor
        self.num_folds_before_pruning = num_folds_before_pruning
        self.is_ddp_backend = False
        self.strikes = 0

    def on_fit_start(self, trainer: Trainer, pl_module: pl.LightningModule) -> None:
        pass  # Not necessary

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking:
            return

        current_score = trainer.callback_metrics.get(self.monitor)

        assert (
            current_score is not None
        ), f"Key {self.monitor} not found in trainer.callback_metrics."

        epoch = pl_module.current_epoch
        should_stop = False

        if not self.is_ddp_backend:
            self._trial.report(current_score.item(), step=epoch)

            # Stopping condition: strikes > num_folds_before_pruning and trial.should_prune()

            if self._trial.should_prune() and self.strikes > 3:
                self.strikes = 0  # Reset strikes
                raise optuna.TrialPruned(
                    f"Trial {self._trial.number} pruned after {trainer.current_epoch} epochs."
                )
            elif self._trial.should_prune() and self.strikes < 3:
                self.strikes += 1
                trainer.should_stop = (
                    trainer.should_stop or should_stop
                )  # Stop training on the current fold
            else:
                return


if __name__ == "__main__":
    pass
