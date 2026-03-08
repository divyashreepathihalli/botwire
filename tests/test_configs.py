"""Tests for botwire configuration system."""

import pytest

from botwire.configs import (
    get_act_config,
    get_base_train_config,
    get_brax_env_config,
    get_d4pg_config,
    get_dataset_config,
    get_diffusion_config,
    get_moma_env_config,
    get_sac_config,
    get_vla_config,
)


def test_get_base_train_config():
    cfg = get_base_train_config()
    assert cfg.mode == "il"
    assert cfg.num_steps > 0
    assert cfg.seed == 42


def test_get_dataset_config():
    cfg = get_dataset_config()
    assert cfg.batch_size > 0
    assert cfg.sequence_length > 0


def test_get_brax_env_config():
    cfg = get_brax_env_config()
    assert cfg.env_name == "ant"
    assert cfg.backend in ("mjx", "generalized", "positional", "spring")


def test_get_moma_env_config():
    cfg = get_moma_env_config()
    assert cfg.robot == "franka_fr3"


def test_get_act_config():
    cfg = get_act_config()
    assert cfg.hidden_dim > 0
    assert cfg.chunk_size > 0
    assert cfg.latent_dim > 0


def test_get_diffusion_config():
    cfg = get_diffusion_config()
    assert cfg.num_diffusion_steps > 0
    assert cfg.num_inference_steps <= cfg.num_diffusion_steps


def test_get_sac_config():
    cfg = get_sac_config()
    assert cfg.learning_rate > 0
    assert cfg.discount < 1.0


def test_get_d4pg_config():
    cfg = get_d4pg_config()
    assert cfg.num_atoms > 0
    assert cfg.v_min < cfg.v_max


def test_config_override():
    cfg = get_base_train_config()
    cfg.num_steps = 999
    assert cfg.num_steps == 999
