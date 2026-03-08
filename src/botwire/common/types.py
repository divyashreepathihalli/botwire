"""Type aliases used throughout botwire.

All public modules share this vocabulary so that policy, environment, and
dataset interfaces remain type-compatible without circular imports.
"""

from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import TypeAlias

# A JAX random key (shape (), dtype uint32[2] or the new key type)
RNGKey: TypeAlias = jax.Array

# A pytree of JAX arrays representing model parameters
Params: TypeAlias = Any

# Structured observation: maps sensor/camera name → array
# e.g. {"image": jnp.ndarray (H, W, 3), "proprio": jnp.ndarray (D,)}
Observation: TypeAlias = dict[str, jax.Array]

# A flat action vector
Action: TypeAlias = jax.Array

# A batched transition for training
# Keys: "observation", "action", "reward", "next_observation", "done"
# Optionally: "discount", "extras"
Batch: TypeAlias = dict[str, jax.Array]

# Metrics returned from update steps
Metrics: TypeAlias = dict[str, float | jax.Array]

# Episode-level data as returned by RLDS/tfds
Episode: TypeAlias = dict[str, Any]

# NumPy array (for pre-processing before JAX conversion)
NpArray: TypeAlias = np.ndarray
