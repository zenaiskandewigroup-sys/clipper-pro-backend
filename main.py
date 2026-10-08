import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from services.whisper_service import transcribe_video
from services.gemini_service import analyze_highlights
from services.ffmpeg_service import crop_video_to_short

load_dotenv()

app = FastAPI(title="ClipperPRO AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.abspath("./uploads")
OUTPUT_DIR = os.path.abspath("./outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: File index.html tidak ditemukan.</h1>"

@app.post("/api/analyze")
async def process_and_analyze(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())[:8]
    saved_filename = f"{file_id}_{file.filename}"
    video_path = os.path.join(UPLOAD_DIR, saved_filename)
    
    with open(video_path, "wb") as buffer:
        buffer.write(await file.read())
        
    try:
        transcript = transcribe_video(video_path)
        highlights = analyze_highlights(transcript)
        
        return {
            "status": "success",
            "video_id": saved_filename,
            "highlights": highlights
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class RenderRequest(BaseModel):
    video_id: str
    start_seconds: int
    end_seconds: int

@app.post("/api/render-clip")
async def render_clip(req: RenderRequest):
    input_video_path = os.path.join(UPLOAD_DIR, req.video_id)
    if not os.path.exists(input_video_path):
        raise HTTPException(status_code=404, detail="File video tidak ditemukan.")
        
    out_filename = f"short_{uuid.uuid4()[:6]}_{req.video_id}"
    output_video_path = os.path.join(OUTPUT_DIR, out_filename)
    
    try:
        crop_video_to_short(
            input_path=input_video_path,
            output_path=output_video_path,
            start_sec=req.start_seconds,
            end_sec=req.end_seconds
        )
        
        return {
            "status": "success",
            "download_url": f"http://127.0.0.1:8000/outputs/{out_filename}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
