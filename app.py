from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from api.admin_api import router as admin_router
from api.enroll_api import router as enroll_router
from api.match_api import router as match_router
from api.media_api import router as media_router
from src.config import FRONTEND_PATH
from src.embed import model_loaded
from src.voice_service import initialize_system


@asynccontextmanager
async def lifespan(app: FastAPI):
    # All state lives in PostgreSQL, so any number of workers can run.
    initialize_system()
    yield


app = FastAPI(
    title="Voice Enrollment and Matching API",
    description="Speaker recognition API backed by PostgreSQL + pgvector.",
    version="2.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=FRONTEND_PATH), name="static")

# Enrolled clips are served from the database by api/media_api.py.
app.include_router(enroll_router)
app.include_router(match_router)
app.include_router(media_router)
app.include_router(admin_router)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/static/console.html")


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok", "speaker_model_loaded": model_loaded()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
