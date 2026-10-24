"""Stateless HTTP API for concurrent synthetic consultation experiments."""

import json
import logging
import os
import socket
from contextlib import asynccontextmanager
from threading import Lock
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from serving.backend import backend_from_environment


LOGGER = logging.getLogger("consultation_api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


class TurnRequest(BaseModel):
    client_id: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=128)
    scenario_id: str = Field(min_length=1, max_length=128)
    patient_answers: int = Field(ge=1, le=5)
    device_type: str = Field(default="unknown", max_length=64)


class Counters:
    def __init__(self):
        self.lock = Lock()
        self.requests = 0
        self.errors = 0
        self.latency_seconds = 0.0

    def observe(self, elapsed, error=False):
        with self.lock:
            self.requests += 1
            self.errors += int(error)
            self.latency_seconds += elapsed

    def render(self):
        with self.lock:
            return (
                "# TYPE consultation_requests_total counter\n"
                f"consultation_requests_total {self.requests}\n"
                "# TYPE consultation_errors_total counter\n"
                f"consultation_errors_total {self.errors}\n"
                "# TYPE consultation_latency_seconds_sum counter\n"
                f"consultation_latency_seconds_sum {self.latency_seconds:.6f}\n"
            )


def create_app(backend=None):
    @asynccontextmanager
    async def lifespan(app):
        if app.state.backend is None:
            app.state.backend = backend_from_environment()
        yield

    app = FastAPI(title="Conversational diagnosis simulation API", lifespan=lifespan)
    app.state.backend = backend
    app.state.replica = os.getenv("HOSTNAME", socket.gethostname())
    app.state.counters = Counters()

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "replica": app.state.replica}

    @app.get("/readyz")
    def readyz():
        if app.state.backend is None:
            raise HTTPException(status_code=503, detail="model is not loaded")
        return {
            "status": "ready",
            "backend": app.state.backend.name,
            "replica": app.state.replica,
        }

    @app.get("/metrics")
    def metrics():
        return Response(app.state.counters.render(), media_type="text/plain")

    @app.post("/v1/simulations/consultations/turn")
    def consultation_turn(payload: TurnRequest, request: Request):
        started = perf_counter()
        error = False
        try:
            result = app.state.backend.predict(
                payload.scenario_id, payload.patient_answers
            )
        except KeyError:
            error = True
            raise HTTPException(status_code=404, detail="unknown synthetic scenario")
        except Exception:
            error = True
            LOGGER.exception("inference failed")
            raise HTTPException(status_code=500, detail="inference failed")
        finally:
            elapsed = perf_counter() - started
            app.state.counters.observe(elapsed, error=error)
        result.update({
            "client_id": payload.client_id,
            "request_id": payload.request_id,
            "device_type": payload.device_type,
            "replica": app.state.replica,
            "latency_ms_server": round(elapsed * 1000, 3),
        })
        LOGGER.info(json.dumps({
            "event": "consultation_turn",
            "client_id": payload.client_id,
            "request_id": payload.request_id,
            "scenario_id": payload.scenario_id,
            "patient_answers": payload.patient_answers,
            "device_type": payload.device_type,
            "replica": app.state.replica,
            "backend": result["backend"],
            "should_stop": result["should_stop"],
            "latency_ms": result["latency_ms_server"],
            "source": request.client.host if request.client else "unknown",
        }, sort_keys=True))
        return result

    return app


app = create_app()
