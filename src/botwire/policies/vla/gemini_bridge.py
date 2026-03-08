"""Gemini Robotics SDK adapter for botwire.

Bridges the Gemini Robotics SDK's Safari model API with the BotwirePolicy
interface, enabling deployment of Gemini Robotics-ER models for VLA inference.

Requires:
    pip install botwire[vla]
    # and enrollment in the Gemini Robotics Trusted Tester Program

Reference:
    https://github.com/google-deepmind/gemini-robotics-sdk
"""

from __future__ import annotations

import base64
import io
from typing import Any

import jax.numpy as jnp
import numpy as np

from botwire.common.abc import RobotInterface
from botwire.common.types import Action, Observation


class GeminiRoboticsAdapter:
    """Adapter that wraps the Gemini Robotics SDK model for botwire inference.

    This adapter connects to either the Gemini Robotics API or a local
    model server (via the Safari SDK) and translates between botwire's
    Observation/Action types and the SDK's request format.

    Args:
        model_id: Gemini Robotics model identifier.
            e.g. "gemini-robotics-er-2.0"
        api_key: Gemini API key. If None, uses GOOGLE_API_KEY env var.
        action_dim: Dimensionality of the robot's action space.
        thinking_budget: Reasoning budget (0 = fast, higher = more reasoning).
        temperature: Sampling temperature for action generation.
        use_local_server: If True, connect to a local Safari SDK model server.
        local_server_url: URL of the local model server.
    """

    def __init__(
        self,
        model_id: str = "gemini-robotics-er-2.0",
        api_key: str | None = None,
        action_dim: int = 7,
        thinking_budget: int = 0,
        temperature: float = 0.0,
        use_local_server: bool = False,
        local_server_url: str = "http://localhost:8080",
    ) -> None:
        self.model_id = model_id
        self.action_dim = action_dim
        self.thinking_budget = thinking_budget
        self.temperature = temperature
        self.use_local_server = use_local_server
        self.local_server_url = local_server_url

        self._client = self._build_client(api_key)

    def _build_client(self, api_key: str | None) -> Any:
        if self.use_local_server:
            return _LocalModelClient(self.local_server_url, self.model_id)

        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError(
                "google-generativeai not installed. Run: pip install botwire[vla]"
            ) from e

        import os

        key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if key:
            genai.configure(api_key=key)
        return genai.GenerativeModel(self.model_id)

    def predict_action(
        self,
        observation: Observation,
        task_description: str,
    ) -> np.ndarray:
        """Run VLA inference and return a predicted action.

        Args:
            observation: Dict with "image" (H, W, 3 uint8) and optionally "proprio".
            task_description: Natural-language task description.

        Returns:
            Predicted action as a NumPy array of shape (action_dim,).
        """
        if self.use_local_server:
            return self._client.predict(observation, task_description)

        # Build Gemini API request
        contents = self._build_request(observation, task_description)
        response = self._client.generate_content(
            contents,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": 256,
            },
        )
        return self._parse_response(response.text)

    def _build_request(
        self,
        observation: Observation,
        task_description: str,
    ) -> list[Any]:
        """Build a Gemini API content list from observation."""
        parts = [f"Task: {task_description}\n\nPredict the robot action."]

        # Add image if available
        if "image" in observation:
            img = np.array(observation["image"])
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)

            # Encode as PNG bytes
            try:
                from PIL import Image as PILImage

                pil_img = PILImage.fromarray(img)
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                img_bytes = buf.getvalue()
            except ImportError:
                img_bytes = img.tobytes()

            import google.generativeai as genai

            parts = [
                genai.types.Part.from_bytes(img_bytes, "image/png"),
                f"Task: {task_description}\n\nPredict the robot action as a list of {self.action_dim} floats.",
            ]

        # Add proprioception
        if "proprio" in observation:
            proprio = np.array(observation["proprio"]).tolist()
            parts.append(f"Current joint state: {proprio}")

        return parts

    def _parse_response(self, response_text: str) -> np.ndarray:
        """Parse the model's text response into an action array."""
        import re

        # Extract list of floats from the response
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", response_text)
        floats = [float(n) for n in numbers[: self.action_dim]]

        if len(floats) < self.action_dim:
            # Pad with zeros if needed
            floats.extend([0.0] * (self.action_dim - len(floats)))

        action = np.array(floats[: self.action_dim], dtype=np.float32)
        return np.clip(action, -1.0, 1.0)


class _LocalModelClient:
    """Client for a locally-running Gemini Robotics Safari SDK model server."""

    def __init__(self, server_url: str, model_id: str) -> None:
        self.server_url = server_url.rstrip("/")
        self.model_id = model_id

    def predict(self, observation: Observation, task_description: str) -> np.ndarray:
        try:
            import requests
        except ImportError as e:
            raise ImportError("Install requests: pip install requests") from e

        payload = {
            "model_id": self.model_id,
            "task": task_description,
            "observation": {
                k: v.tolist() if hasattr(v, "tolist") else v
                for k, v in observation.items()
                if k != "image"  # images require special handling
            },
        }
        response = requests.post(f"{self.server_url}/predict", json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        return np.array(data["action"], dtype=np.float32)
