import os
from functools import cache

from fastapi import APIRouter, Depends, UploadFile
from groq import AsyncGroq

from app.schemas import Transcription

router = APIRouter()


@cache
def get_groq() -> AsyncGroq:
    return AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))


@router.post("/transcribe")
async def transcribe(
    audio: UploadFile,
    client: AsyncGroq = Depends(get_groq),
) -> Transcription:
    """Call OpenAI's Whisper large v3 via Groq to transcribe an audio file to text."""
    result = await client.audio.transcriptions.create(
        file=(audio.filename, await audio.read()),
        model="whisper-large-v3",
    )
    return Transcription(text=result.text)
