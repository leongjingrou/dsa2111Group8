import os
from pathlib import Path
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from typing import Literal 


# Root of the repo: llm_service/app/config.py 
REPO_ROOT = Path(__file__).resolve().parents[2]

# Default paths.
DEFAULT_CONDITIONS = REPO_ROOT / "knowledge_base" / "data" / "conditions.json"
DEFAULT_SYMPTOMS = Path(__file__).resolve().parents[1] / "data" / "symptoms.json"

# Environment variables override the defaults (used in Docker and the cloud VM)
CONDITIONS_PATH = Path(os.getenv("CONDITIONS_PATH", DEFAULT_CONDITIONS))
SYMPTOMS_PATH = Path(os.getenv("SYMPTOMS_PATH", DEFAULT_SYMPTOMS))

#LLM api
ENV_FILE = Path(__file__).resolve().parents[2] / ".env" 

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")
 
    # Accepts either LLM_API_KEY or GROQ_API_KEY (Role 6's existing name)
    llm_api_key: str = Field(
        default="", validation_alias=AliasChoices("LLM_API_KEY", "GROQ_API_KEY")
    )
    llm_base_url: str | None = None   # e.g. https://api.groq.com/openai/v1 ; None = OpenAI
    llm_model: str = ""
    llm_timeout_s: float = 30
    llm_max_retries: int = 2
 
    # "json_object": works on most OpenAI-compatible providers (Groq, HF, ...)
    # "schema":      OpenAI strict structured outputs (needs a model that supports it)
    llm_structured_mode: Literal["json_object", "schema"] = "json_object"
 
 
settings = Settings()
