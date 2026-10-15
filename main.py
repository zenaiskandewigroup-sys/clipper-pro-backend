import os
import subprocess
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import yt_dlp

# --- SISTEM KEAMANAN API KEY ---
# Mengambil kunci rahasia dari Environment Variables Render
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GROQ_API_KEY or not GEMINI_API_KEY:
    print("WARNING: API KEY GROQ ATAU GEMINI BELUM DIPASANG DI ENVIRONMENT RENDER!")
# -------------------------------

app = FastAPI()

# Izinkan Vercel ngobrol sama Render
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Buat folder output kalau belum ada
os.makedirs("outputs", exist_ok=True)

# Format Data dari Vercel
class YTAnalyzeRequest(BaseModel):
    video_url: str

class YTRenderRequest(BaseModel):
    video_url: str
    start_seconds: int
    end_seconds: int

# --- ENDPOINT 1: ANALISIS YOUTUBE ---
@app.post("/api/analyze-youtube")
async def analyze_youtube(req: YTAnalyzeRequest):
    try:
        # 1. Sedot audio saja dari YouTube (Biar hemat RAM Render)
        audio_filename = f"outputs/audio_{uuid.uuid4().hex}.m4a"
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': audio_filename,
            'quiet': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([req.video_url])
            
        # ==========================================
        # MASUKIN SCRIPT GROQ & GEMINI LU DI SINI
        # Pastikan saat manggil API, lu pake variabel:
        # GROQ_API_KEY dan GEMINI_API_KEY
        # ==========================================
        
        # Hapus audio setelah ditranskrip biar hardisk Render gak penuh
        if os.path.exists(audio_filename):
            os.remove(audio_filename)

        # Output Dummy (Ganti pakai hasil output dari AI Gemini lu nanti)
        return {
            "message": "Sukses",
            "highlights": [
                {
                    "title": "Momen Epik 1",
                    "hook_text": "Ini adalah kutipan viral pertama...",
                    "start_seconds": 30, # Contoh: mulai di detik 30
                    "end_seconds": 90,   # Contoh: selesai di detik 90
                    "suggested_titles": ["Judul Viral A", "Judul Viral B"]
                }
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- ENDPOINT 2: POTONG & RENDER VIDEO ---
@app.post("/api/render-youtube-clip")
async def render_youtube_clip(req: YTRenderRequest):
    try:
        output_filename = f"outputs/clip_{uuid.uuid4().hex}.mp4"
        
        # 1. Dapatkan Link Streaming Langsung dari YouTube (Bypass Download Penuh)
        ydl_opts = {'format': 'best', 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(req.video_url, download=False)
            direct_url = info['url']
            
        # 2. Potong langsung dari Cloud pakai FFmpeg (Hanya butuh RAM super kecil!)
        # -ss = mulai, -to = akhir, -vf crop = potong vertikal 9:16
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(req.start_seconds),
            "-to", str(req.end_seconds),
            "-i", direct_url,
            "-vf", "crop=ih*9/16:ih", 
            "-c:a", "copy",
            output_filename
        ]
        
        subprocess.run(cmd, check=True)
        
        # Kembalikan link download ke Vercel
        return {"download_url": f"/{output_filename}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- ENDPOINT DOWNLOAD ---
@app.get("/outputs/{filename}")
async def download_file(filename: str):
    file_path = os.path.join("outputs", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4", filename=filename)
    raise HTTPException(status_code=404, detail="File tidak ditemukan")
