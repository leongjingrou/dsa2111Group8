"""Build the patient API using one shared, mandatory safety workflow."""

from dotenv import load_dotenv

from llm_service.orchestrated_main import create_orchestrated_app

from .api import diagnose, transcribe

load_dotenv()


def create_app(*args, **kwargs):
    app = create_orchestrated_app(*args, **kwargs)
    app.include_router(diagnose.router, prefix="/api")
    app.include_router(transcribe.router, prefix="/api")
    return app


app = create_app()
