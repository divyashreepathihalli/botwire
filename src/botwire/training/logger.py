"""Metrics logger supporting console, WandB, and TensorBoard."""

from __future__ import annotations

from typing import Any

from absl import logging


class MetricsLogger:
    """Unified logger for training metrics.

    Args:
        log_dir: Directory for TensorBoard logs.
        use_wandb: Enable Weights & Biases logging.
        wandb_project: WandB project name.
        wandb_entity: WandB entity (team or user).
        use_tensorboard: Enable TensorBoard logging.
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        use_wandb: bool = False,
        wandb_project: str = "botwire",
        wandb_entity: str = "",
        use_tensorboard: bool = False,
    ) -> None:
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self._tb_writer = None

        if use_wandb:
            self._init_wandb(wandb_project, wandb_entity)

        if use_tensorboard:
            self._init_tensorboard(log_dir)

    def _init_wandb(self, project: str, entity: str) -> None:
        try:
            import wandb

            wandb.init(project=project, entity=entity or None)
            logging.info("WandB initialized: project='%s'", project)
        except ImportError:
            logging.warning("wandb not installed; disabling WandB logging.")
            self.use_wandb = False

    def _init_tensorboard(self, log_dir: str) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._tb_writer = SummaryWriter(log_dir=log_dir)
            logging.info("TensorBoard writer initialized at '%s'", log_dir)
        except ImportError:
            try:
                import tensorflow as tf

                self._tb_writer = tf.summary.create_file_writer(log_dir)
            except ImportError:
                logging.warning("Neither PyTorch nor TF TensorBoard available.")
                self.use_tensorboard = False

    def log(
        self,
        metrics: dict[str, Any],
        step: int,
        prefix: str = "",
    ) -> None:
        """Log metrics to all enabled backends.

        Args:
            metrics: Dict of metric name → scalar value.
            step: Current training step.
            prefix: Optional prefix for all metric names.
        """
        prefixed = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}

        # Console
        parts = [f"{k}={float(v):.4f}" for k, v in prefixed.items()]
        logging.info("step=%d  %s", step, "  ".join(parts))

        if self.use_wandb:
            try:
                import wandb

                wandb.log({k: float(v) for k, v in prefixed.items()}, step=step)
            except Exception as e:
                logging.warning("WandB logging failed: %s", e)

        if self.use_tensorboard and self._tb_writer is not None:
            try:
                for k, v in prefixed.items():
                    self._tb_writer.add_scalar(k, float(v), global_step=step)
            except Exception as e:
                logging.warning("TensorBoard logging failed: %s", e)

    def close(self) -> None:
        if self._tb_writer is not None:
            try:
                self._tb_writer.close()
            except Exception:
                pass

        if self.use_wandb:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
