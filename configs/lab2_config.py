from pydantic import BaseModel, Field
from pathlib import Path

class Lab2Config(BaseModel):
    data_path: Path = Path("data/raw/arrhythmia.data")
    chunk_size: int = Field(default=64, gt=0, le=1000)
    test_size: float = Field(default=0.2, gt=0, lt=1)
    random_state: int = Field(default=42, ge=0)
    contamination: float = Field(default=0.15, gt=0, lt=0.5)
    n_estimators: int = Field(default=100, ge=10, le=500)
    sgd_alpha: float = Field(default=0.0001, gt=0)
    epochs: int = Field(default=10, ge=1, le=100)
