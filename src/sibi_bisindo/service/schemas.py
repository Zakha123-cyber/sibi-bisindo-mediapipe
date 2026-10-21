from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class Timings(BaseModel):
    detector: float = Field(ge=0)
    classifier: float = Field(ge=0)
    total: float = Field(ge=0)


class Preprocessing(BaseModel):
    method: str
    attempts: int = Field(ge=0)
    recovered: bool
    input_was_mirrored: bool
    coordinates: Literal['unmirrored_normalized_xyz']


class Prediction(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    language: Literal['sibi','bisindo']
    model_id: str
    status: str
    detection_status: str
    label: str | None
    score: float | None = Field(ge=0,le=1)
    score_type: str | None
    hand_count: int = Field(ge=0,le=2)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    preprocessing: Preprocessing
    timings_ms: Timings
    landmarks: list[list[list[float]]] | None
