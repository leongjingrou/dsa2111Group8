from typing import Literal

from pydantic import BaseModel

Role = Literal["user", "bot"]
DiagnosticStatus = Literal["asking", "complete", "urgent"]


class Message(BaseModel):
    role: Role
    text: str


class DiagnoseRequest(BaseModel):
    messages: list[Message]


class Diagnosis(BaseModel):
    status: DiagnosticStatus
    reply: str


class Transcription(BaseModel):
    text: str
