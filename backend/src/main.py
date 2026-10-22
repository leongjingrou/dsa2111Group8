from dotenv import load_dotenv
from fastapi import FastAPI

from api import diagnose, transcribe

load_dotenv()

app = FastAPI()
app.include_router(diagnose.router, prefix="/api")
app.include_router(transcribe.router, prefix="/api")
