import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_lite6_example() -> dict:
    """Creates a random input example for the Lite6 station policy.

    Matches the keys produced by the repack transform in ``Lite6Station``:
    a base (exterior) camera, a wrist camera, and a 6-dim proprioceptive state.
    """
    return {
        "observation/state": np.random.rand(6),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class Lite6Inputs(transforms.DataTransformFn):
    """Converts inputs into the model's expected format. Used for both training and inference.

    The Lite6 station has 6-dim joint-velocity actions and two cameras: one base/exterior view and one
    wrist view. pi0.5 always expects three image slots, so we pad the unused ``right_wrist_0_rgb``
    slot with a zero image and mask it out.
    """

    # Determines which model will be used. Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W); gets skipped for policy inference.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad the non-existent second wrist image with a zero-array of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for flow models (pi0 / pi0.5), not pi0-FAST.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Actions are only available during training. They get padded to the model action
        # dimension downstream by PadStatesAndActions, so no need to pad here.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class Lite6Outputs(transforms.DataTransformFn):
    """Converts model outputs back to the dataset-specific format. Used for inference only."""

    def __call__(self, data: dict) -> dict:
        # The model outputs actions padded to its action dimension (32). The Lite6 station uses 6-dim
        # joint-velocity actions, so we return only the first 6 dims.
        return {"actions": np.asarray(data["actions"][..., :6])}
