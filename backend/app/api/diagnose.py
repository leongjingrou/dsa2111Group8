from fastapi import APIRouter

from app.schemas import DiagnoseRequest, Diagnosis

router = APIRouter()


@router.post("/diagnose")
async def diagnose(req: DiagnoseRequest) -> Diagnosis:
    return Diagnosis(status="asking", reply="To be implemented")
