"""Tests for botwire.common module."""

import jax
import jax.numpy as jnp
import pytest

from botwire.common.types import Action, Batch, Observation
from botwire.common.utils import (
    count_parameters,
    format_params,
    get_device,
    seed_everything,
    tree_stack,
    tree_unstack,
)


def test_seed_everything():
    rng = seed_everything(42)
    assert rng.shape == (2,)  # PRNGKey shape


def test_get_device():
    device = get_device("cpu")
    assert device is not None


def test_get_device_auto():
    device = get_device("auto")
    assert device is not None


def test_count_parameters():
    params = {"w": jnp.ones((10, 5)), "b": jnp.ones((5,))}
    assert count_parameters(params) == 55


def test_format_params():
    params = {"w": jnp.ones((1000, 1000))}
    s = format_params(params)
    assert "M" in s or "K" in s or s.isdigit()


def test_tree_stack():
    trees = [{"a": jnp.array([1.0])}, {"a": jnp.array([2.0])}]
    stacked = tree_stack(trees, axis=0)
    assert stacked["a"].shape == (2, 1)


def test_tree_unstack():
    stacked = {"a": jnp.array([[1.0], [2.0]])}
    trees = tree_unstack(stacked)
    assert len(trees) == 2
    assert float(trees[0]["a"][0]) == pytest.approx(1.0)
