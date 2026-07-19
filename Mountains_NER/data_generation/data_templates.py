from typing import List
from pydantic import BaseModel, Field


class MountainRecord(BaseModel):
    text: str = Field(
        ...,
        description="The complete raw sentence containing one or more mountain names."
    )

    markers: List[List[int]] = Field(
        ...,
        description="List of character index pairs [start_char_idx, end_char_idx] pointing exactly to the mountain names."
    )

    entity_names: List[str] = Field(
        ...,
        description="The exact text string(s) of the mountain names extracted by the spans (e.g., ['Mount Rainier'])."
    )


class MountainBatch(BaseModel):
    records: List[MountainRecord] = Field(
        ...,
        description="A collection of annotated mountain sentences."
    )