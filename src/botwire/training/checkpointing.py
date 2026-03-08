"""Orbax-based checkpoint management for botwire.

Handles saving and restoring JAX pytrees (model params, optimizer states)
using Orbax, which is the GDM-recommended checkpointing library for JAX.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import jax
import orbax.checkpoint as ocp
from absl import logging


class CheckpointManager:
    """Manages multiple checkpoints with a configurable keep policy.

    Args:
        directory: Root directory for checkpoints.
        max_to_keep: Number of most recent checkpoints to retain.
        save_interval_steps: Save every N steps (informational; callers decide when to save).
    """

    def __init__(
        self,
        directory: str | Path,
        max_to_keep: int = 3,
        save_interval_steps: int = 10_000,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.save_interval_steps = save_interval_steps

        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            save_interval_steps=save_interval_steps,
        )
        self._manager = ocp.CheckpointManager(
            str(self.directory),
            checkpointers=ocp.PyTreeCheckpointer(),
            options=options,
        )

    def save(self, step: int, state: Any) -> None:
        """Save a checkpoint at the given step.

        Args:
            step: Training step number (used as checkpoint ID).
            state: Arbitrary JAX pytree (e.g. {"params": ..., "opt_state": ...}).
        """
        self._manager.save(step, state)
        logging.info("Saved checkpoint at step %d to '%s'", step, self.directory)

    def restore(self, step: int | None = None, target: Any = None) -> Any:
        """Restore a checkpoint.

        Args:
            step: Specific step to restore. None = latest available.
            target: Pytree with the same structure as the saved state.
                    Required for Orbax to know the expected shapes.

        Returns:
            Restored state pytree.
        """
        if step is None:
            step = self._manager.latest_step()
            if step is None:
                raise FileNotFoundError(
                    f"No checkpoints found in '{self.directory}'."
                )
        state = self._manager.restore(step, items=target)
        logging.info("Restored checkpoint from step %d", step)
        return state

    def latest_step(self) -> int | None:
        """Return the step number of the most recent checkpoint, or None."""
        return self._manager.latest_step()

    def all_steps(self) -> list[int]:
        """Return a sorted list of all available checkpoint steps."""
        return sorted(self._manager.all_steps())

    def wait_until_finished(self) -> None:
        """Block until all async checkpoint operations complete."""
        self._manager.wait_until_finished()

    def close(self) -> None:
        """Flush pending writes and release resources."""
        self._manager.close()

    def __enter__(self) -> "CheckpointManager":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
