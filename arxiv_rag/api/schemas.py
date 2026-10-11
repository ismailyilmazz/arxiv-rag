from typing import Literal, Optional

from pydantic import BaseModel, Field


class SearchIn(BaseModel):
    query: str = Field(min_length=2, max_length=300)
    k: int = Field(10, ge=1, le=30)


class PlanIn(BaseModel):
    topic: str = Field(min_length=2, max_length=300)
    type: Literal["survey", "proposal", "synthesis"] = "survey"
    lang: Literal["en", "tr"] = "en"
    seeds: list[str] = Field(default_factory=list, max_length=5)
    focus: Optional[str] = Field(None, max_length=100)
    force: bool = False


class DocumentIn(BaseModel):
    plan_id: str = Field(min_length=1, max_length=40)
    removed_ids: list[str] = Field(default_factory=list, max_length=30)
