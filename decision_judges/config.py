"""Study and pricing configuration models with loaders."""

import tomllib
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field

from decision_judges.types import Usage

DEFAULT_CASCADE = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]


class LayaModel(BaseModel):
    """Reference to the open-weights Laya model on the Hugging Face Hub."""

    repo_id: str = "convaiinnovations/laya"
    revision: str


class Models(BaseModel):
    """Model identifiers used across the study."""

    agent: str
    user_sim: str
    llm_cheap: str
    llm_strong: str
    jev: str = "jev-1.13.0"
    laya: LayaModel


class Thresholds(BaseModel):
    """Decision thresholds used by the judge cascade."""

    cascade: list[float] = Field(default_factory=lambda: list(DEFAULT_CASCADE))


G2_DEFAULT_REPEATS: dict[str, int] = {
    "jev": 5,
    "laya_base": 5,
    "laya_ft": 5,
    "llm_cheap": 1,
    "llm_strong": 1,
    "code": 1,
}


class G2Config(BaseModel):
    """Per-judge repeat counts for the high-volume G2 step gate."""

    repeats: dict[str, int] = Field(default_factory=lambda: dict(G2_DEFAULT_REPEATS))


class StudyConfig(BaseModel):
    """Top-level configuration for a study run."""

    models: Models
    repeats: int = 5
    concurrency: dict[str, int]
    spend_caps: dict[str, float]
    thresholds: Thresholds = Field(default_factory=Thresholds)
    g2: G2Config = Field(default_factory=G2Config)
    seed: int = 7
    llm_base_url: str


class ModelPrice(BaseModel):
    """Per-million-token pricing for a single model."""

    input_per_mtok: float
    output_per_mtok: float


class UnpricedModel(KeyError):
    """Raised when a cost is requested for a model absent from the table."""


class PricingTable(BaseModel):
    """Dated table of model prices used to cost token usage."""

    effective_date: date
    source: str
    models: dict[str, ModelPrice]

    def cost(self, model_id: str, usage: Usage) -> float:
        """Return the USD cost of usage for a model priced per million tokens."""
        try:
            price = self.models[model_id]
        except KeyError as exc:
            raise UnpricedModel(model_id) from exc
        return (
            usage.input_tokens * price.input_per_mtok + usage.output_tokens * price.output_per_mtok
        ) / 1_000_000


def _read_toml(path: Path | str) -> dict[str, object]:
    """Parse a TOML file into a dictionary."""
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def load_study(path: Path | str) -> StudyConfig:
    """Load and validate a study configuration from a TOML file."""
    return StudyConfig.model_validate(_read_toml(path))


def load_pricing(path: Path | str) -> PricingTable:
    """Load and validate a pricing table from a TOML file."""
    return PricingTable.model_validate(_read_toml(path))
