"""Dataset configuration using ml_collections."""

import ml_collections


def get_dataset_config() -> ml_collections.ConfigDict:
    """Return the default RLDS dataset configuration."""
    config = ml_collections.ConfigDict()

    # Name of a tensorflow_datasets dataset (Open X-Embodiment or custom)
    config.name = "fractal20220817_data"
    config.split = "train"

    # Data loading
    config.batch_size = 64
    config.sequence_length = 16          # number of timesteps per sample
    config.action_horizon = 4            # prediction horizon for IL policies
    config.shuffle_buffer = 10_000
    config.num_parallel_reads = 8
    config.num_parallel_calls = 8
    config.prefetch = 4

    # Normalization (computed over training set)
    config.normalize_observations = True
    config.normalize_actions = True

    # Image settings (if dataset has image observations)
    config.image_keys = ("image",)       # observation keys that are images
    config.image_size = (224, 224)       # resize target (H, W)
    config.augment_images = True

    # Local RLDS path (used when name="local")
    config.local_path = ""

    return config
