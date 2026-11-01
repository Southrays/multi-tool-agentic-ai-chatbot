from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=10000,
        description="The message sent by the user",
    )
    thread_id: str | None = None
    model: str = Field(
        default="gemini-3.8-flash",
        description="The Gemini model to use",
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message cannot be empty.")
        return value


class ChatResponse(BaseModel):
    response: str
    thread_id: str


class ChatStreamEvent(BaseModel):
    thread_id: str | None = None
    token: str | None = None
    done: bool | None = None
    error: str | None = None