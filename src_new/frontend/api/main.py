from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import tempfile
import os
import shutil
from inference import run_inference

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    start_time: float = Form(0.0),
    end_time: float = Form(-1.0),
):
    # Save uploaded file to a temp location
    suffix = os.path.splitext(file.filename)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = run_inference(
            audio_path=tmp_path,
            start_time=start_time,
            end_time=end_time if end_time > 0 else None,
        )
        return JSONResponse(content=result)
    finally:
        os.remove(tmp_path)


@app.get("/health")
def health():
    return {"status": "ok"}
