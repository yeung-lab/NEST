# evaluation/models/response_formats.py
"""Structured response formats for different question types"""

from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional, Union
from enum import Enum

class QuestionType(str, Enum):
    """Supported question types"""
    MULTIPLE_CHOICE = "multiple_choice"
    OPEN_QUESTION = "open_question"

class MultipleChoiceResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "additionalProperties": False,
        "required": ["answer", "reasoning"]
    })

    answer: str = Field(
        description="The selected answer letter (A, B, C, D, etc.)"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="One line explanation of why this answer was chosen"
    )

class OpenQuestionResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "additionalProperties": False,
        "required": ["answer", "reasoning"]
    })

    answer: str = Field(
        description="The main answer to the question"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="One line explanation of the reasoning behind the answer"
    )

class EvaluationResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})

    is_correct: bool = Field(
        description="Whether the answer is correct or not"
    )
    score: float = Field(
        description="Score from 0.0 to 1.0"
    )
    reasoning: str = Field(
        description="Explanation of the evaluation"
    )

def get_response_format(question_type: str) -> BaseModel:
    """Get the appropriate response format for a question type"""
    if question_type == "multiple_choice":
        return MultipleChoiceResponse
    elif question_type == "open_question":
        return OpenQuestionResponse
    else:
        raise ValueError(f"Unknown question type: {question_type}")

def get_evaluation_format() -> BaseModel:
    """Get the response format for LLM-as-a-judge evaluations"""
    return EvaluationResponse
