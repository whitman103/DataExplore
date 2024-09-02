from typing import Tuple, List
from pydantic import BaseModel, ConfigDict, field_validator
import numpy as np


class Keyframe(BaseModel):
    domain_start: float = 0
    domain_end: float = 1
    target_range: Tuple

    @property
    def domain_range(self) -> float:
        return self.domain_end - self.domain_start


class FloatKeyframe(Keyframe):
    target_range: Tuple[float, float]


class ThreeDKeyFrame(Keyframe):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    target_range: Tuple[np.ndarray, np.ndarray]

    @field_validator("target_range", mode="before")
    @classmethod
    def convert_to_np(cls, in_value: Tuple):
        return tuple([np.array(x) for x in in_value])


class LinearInterpolater:
    def calculate_value(
        self, keyframe: Keyframe, current_control_value: float
    ) -> float:
        normalized_control = (
            current_control_value - keyframe.domain_start
        ) / keyframe.domain_range

        return keyframe.target_range[0] * (
            1 - normalized_control
        ) + keyframe.target_range[1] * (normalized_control)


class AnimatedPoint(BaseModel):
    frames: List[Keyframe]


if __name__ == "__main__":
    test_frame = ThreeDKeyFrame(
        domain_start=0, domain_end=5, target_range=([0, 0, 0], [1, 1, 1])
    )

    interp = LinearInterpolater()
