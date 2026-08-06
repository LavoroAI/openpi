import dataclasses
import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_humanoid_navigate_example() -> dict:
    """Creates a random input example for the HumanoidNavigate policy.

    Matches the keys produced by the repack transform in ``HumanoidNavigate``:
    a base camera image and 6-dim action vector.
    """
    return {
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "navigate to goal",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class HumanoidNavigateInputs(transforms.DataTransformFn):
    """Converts inputs into the model's expected format. Used for both training and inference.

    The humanoid_navigate setup has 6-dim action vector [x, z, u, w, vel, curvature] and one base camera.
    pi0.5 always expects three image slots, so we pad the unused wrist camera slots with zero images
    and mask them out.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])

        inputs = {
            "state": data.get("observation/state", np.zeros(0, dtype=np.float32)),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class HumanoidNavigateOutputs(transforms.DataTransformFn):
    """Converts model outputs back to dataset-specific format."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :6])}
