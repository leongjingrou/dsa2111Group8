import os
from functools import cache

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from ..schemas import Transcription

router = APIRouter()


@cache
def get_groq():
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(503, "Speech transcription is not configured")
    from groq import AsyncGroq

    return AsyncGroq(api_key=os.environ["GROQ_API_KEY"])


@router.post("/transcribe")
async def transcribe(
    audio: UploadFile,
    client=Depends(get_groq),
) -> Transcription:
    """Call OpenAI's Whisper large v3 via Groq to transcribe an audio file to text."""
    result = await client.audio.transcriptions.create(
        file=(audio.filename, await audio.read()),
        model="whisper-large-v3",
    )
    return Transcription(text=result.text)
