import logging
import os
from pathlib import Path
from typing import Dict, Optional

import wandb
from patronum.logging.mask import ILogger

logger = logging.getLogger(__name__)


class WANDBLogger(ILogger):
    """
    Weights and biases logger. See <a href="https://docs.wandb.ai">here</a> for more details.
    """

    experiment = None
    save_dir = str(Path(os.getcwd()) / ".wandb")
    offset_step = 0
    sync_step = True
    prefix = ""
    log_checkpoint = False

    @classmethod
    def init_experiment(
        cls,
        experiment_name,
        project_name,
        api: Optional[str] = None,
        notes=None,
        tags=None,
        entity=None,
        save_dir: Optional[str] = None,
        offline: Optional[bool] = False,
        _id: Optional[str] = None,
        log_checkpoint: Optional[bool] = False,
        sync_step: Optional[bool] = True,
        prefix: Optional[str] = "",
        notebook: Optional[str] = None,
        **kwargs,
    ):
        if offline:
            os.environ["WANDB_MODE"] = "dryrun"
        if api is not None:
            os.environ["WANDB_API_KEY"] = api
        os.environ["WANDB_RESUME"] = "allow"
        os.environ["WANDB_RUN_ID"] = wandb.util.generate_id() if _id is None else _id
        os.environ["WANDB_NOTEBOOK_NAME"] = notebook if notebook else "thebatai"

        if wandb.run is not None:
            cls.end_run()

        cls.experiment = wandb.init(
            resume=sync_step,
            name=experiment_name,
            dir=save_dir,
            project=project_name,
            notes=notes,
            tags=tags,
            entity=entity,
            **kwargs,
        )

        cls.offset_step = cls.experiment.step
        cls.prefix = prefix
        cls.sync_step = sync_step
        cls.log_checkpoint = log_checkpoint

        return cls(tracking_uri=cls.experiment.url)

    @classmethod
    def end_run(cls):
        if cls.experiment is not None:
            # Global step saving for future resuming
            cls.offset_step = cls.experiment.step
            # Send all checkpoints to WB server
            if cls.log_checkpoint:
                wandb.save(os.path.join(cls.save_dir, "*ckpt"))
            cls.experiment.finish()

    def log_metrics(self, metrics, step, **kwargs):
        assert self.experiment is not None, "Initialize experiment first by calling `WANDBLogger.init_experiment(...)`"
        metrics = {f"{self.prefix}{k}": v for k, v in metrics.items()}
        if self.sync_step and step + self.offset_step < self.experiment.step:
            logger.warning("Trying to log at a previous step. Use `sync_step=False`")
        if self.sync_step:
            self.experiment.log(metrics, step=(step + self.offset_step) if step is not None else None)
        elif step is not None:
            self.experiment.log({**metrics, "step": step + self.offset_step}, **kwargs)
        else:
            self.experiment.log(metrics)

    def log_params(self, params):
        assert self.experiment is not None, "Initialize experiment first by calling `WANDBLogger.init_experiment(...)`"
        self.experiment.config.update(params, allow_val_change=True)

    def log_artifacts(self, artifacts):
        raise NotImplementedError()


def _format_metrics(dct: Dict):
    return " | ".join([f"{k}: {float(dct[k])}" for k in sorted(dct.keys())])


class ConsoleLogger(ILogger):
    """Console logger for parameters and metrics.
    Output the metric into the console during experiment.

    Args:
        log_hparams: boolean flag to print all hparams to the console (default: False)

    .. note::
        This logger is used by default by all Runners.
    """

    def __init__(self, log_hparams: bool = False):
        super().__init__(tracking_uri=None)
        self._log_hparams = log_hparams

    @classmethod
    def init_experiment(cls, **kwargs):
        return cls(**kwargs)

    def log_params(self, hparams: Dict) -> None:
        """Logs hyperparameters to the console."""
        if self._log_hparams:
            print(f"Hparams: {hparams}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step,
        total_steps=None,
    ) -> None:
        """Logs loader and epoch metrics to stdout."""
        prefix = f"{str(step)}/{str(total_steps)} ### " if total_steps is not None else f"{str(step)} ### "
        msg = prefix + _format_metrics(metrics)
        print(msg)

    @classmethod
    def end_run(cls):
        pass

    def log_artifacts(self, artifacts, **kwargs):
        pass


__all__ = ["ConsoleLogger", "WANDBLogger"]
