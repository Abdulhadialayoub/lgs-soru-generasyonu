from pydantic import BaseModel
from typing import Optional

class Question(BaseModel):
    id: Optional[int] = None
    year: int
    question_number: int
    question_text: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    correct_option: str
    topic: str

class QuestionResponse(BaseModel):
    success: bool
    message: str
    data: Optional[list] = None
    total_count: Optional[int] = None