from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_service.api_schema import ConsultationResponse

# Retain the existing import name while both routes share one response schema.
Diagnosis = ConsultationResponse

Role = Literal["user", "bot"]
DiagnosticStatus = Literal["asking", "complete", "urgent"]


class Message(BaseModel):
    role: Role
    text: str = Field(min_length=1, max_length=2000)


class DiagnoseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: list[Message] = Field(default_factory=list, max_length=128)
    patient_message: str | None = Field(
        default=None, min_length=1, max_length=2000, pattern=r"\S"
    )
    conversation_id: str | None = Field(default=None, min_length=1, max_length=64)
    request_id: str | None = Field(default=None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def one_message_source(self):
        if bool(self.messages) == (self.patient_message is not None):
            raise ValueError("Provide either patient_message or messages")
        if self.messages and self.messages[-1].role != "user":
            raise ValueError("The last message must be from the patient")
        if (
            self.messages
            and self.conversation_id is None
            and sum(m.role == "user" for m in self.messages) != 1
        ):
            raise ValueError(
                "Return conversation_id on subsequent turns; history is not trusted as screening state"
            )
        return self


class Transcription(BaseModel):
    text: str
