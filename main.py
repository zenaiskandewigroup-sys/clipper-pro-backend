import os
import subprocess
import uuid
import json
import cv2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import yt_dlp
from groq import Groq
from google import genai
from google.genai import types

# --- SISTEM KEAMANAN API KEY & COOKIES ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
YOUTUBE_COOKIES = os.getenv("YOUTUBE_COOKIES")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("outputs", exist_ok=True)

# Tulis cookies rahasia ke dalam file sistem Render
if YOUTUBE_COOKIES:
    with open("cookies.txt", "w") as f:
        f.write(YOUTUBE_COOKIES)

class YTAnalyzeRequest(BaseModel):
    video_url: str

class YTRenderRequest(BaseModel):
    video_url: str
    start_seconds: int
    end_seconds: int

# ==========================================
# 1. LOGIKA GROQ WHISPER
# ==========================================
def transcribe_video(audio_path: str) -> str:
    if not GROQ_API_KEY:
        raise Exception("API Key Groq belum dipasang!")
        
    client = Groq(api_key=GROQ_API_KEY)
    compressed_audio = audio_path + ".mp3"
    
    subprocess.run([
        "ffmpeg", "-y", "-i", audio_path, 
        "-vn", "-c:a", "libmp3lame", "-b:a", "32k", compressed_audio
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    with open(compressed_audio, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(compressed_audio), audio_file.read()),
            model="whisper-large-v3",
            response_format="verbose_json",
        )
        
    if os.path.exists(compressed_audio): os.remove(compressed_audio)
        
    formatted_transcript = []
    segments = getattr(transcription, 'segments', [])
    if not segments and isinstance(transcription, dict):
        segments = transcription.get('segments', [])
        
    if segments:
        for segment in segments:
            start_time = segment['start'] if isinstance(segment, dict) else segment.start
            end_time = segment['end'] if isinstance(segment, dict) else segment.end
            text = segment['text'] if isinstance(segment, dict) else segment.text
            start_min, start_sec = int(start_time // 60), int(start_time % 60)
            end_min, end_sec = int(end_time // 60), int(end_time % 60)
            timestamp = f"[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}]"
            formatted_transcript.append(f"{timestamp} {text.strip()}")
    else:
        text = transcription.text if hasattr(transcription, 'text') else transcription.get('text', '')
        formatted_transcript.append(f"[00:00 - End] {text}")
        
    return "\n".join(formatted_transcript)

# ==========================================
# 2. LOGIKA GEMINI
# ==========================================
def analyze_highlights(transcript_text: str):
    if not GEMINI_API_KEY:
        raise Exception("API Key Gemini belum dipasang!")
        
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"""
    Kamu adalah Editor Video TikTok Profesional.
    Analisis transkrip ini dan pilih 1-3 momen viral (durasi ideal 30-60 detik).
    Return HANYA format array JSON valid persis seperti ini:
    [
      {{
        "title": "Judul",
        "start_seconds": 15,
        "end_seconds": 60,
        "reason": "Alasan",
        "hook_text": "Kutipan",
        "suggested_titles": ["Judul 1"]
      }}
    ]
    Transkrip:
    {transcript_text}
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    
    raw_json = response.text.strip()
    if raw_json.startswith("```json"): raw_json = raw_json[7:]
    elif raw_json.startswith("```"): raw_json = raw_json[3:]
    if raw_json.endswith("```"): raw_json = raw_json[:-3]
        
    try:
        return json.loads(raw_json.strip())
    except json.JSONDecodeError:
        raise Exception("Gagal membaca respons dari Gemini.")

# ==========================================
# 3. LOGIKA OPENCV & FFMPEG
# ==========================================
def crop_video_to_short(input_url: str, output_path: str, start_sec: int, end_sec: int):
    duration = end_sec - start_sec
    uid = str(uuid.uuid4())[:6]
    temp_clip = f"outputs/temp_raw_clip_{uid}.mp4"
    temp_tracked = f"outputs/temp_tracked_{uid}.mp4"
    
    cmd_cut = [
        "ffmpeg", "-y", "-ss", str(start_sec), "-i", input_url,
        "-t", str(duration), "-c:v", "copy", "-c:a", "copy", temp_clip
    ]
    subprocess.run(cmd_cut, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    cap = cv2.VideoCapture(temp_clip)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps: fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    target_width = int(height * 9 / 16)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_tracked, fourcc, fps, (target_width, height))
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    last_center_x = width // 2
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            x, y, w, h = faces[0]
            current_center_x = x + w // 2
            last_center_x = int(last_center_x * 0.9 + current_center_x * 0.1)
            
        x1 = max(0, last_center_x - target_width // 2)
        x2 = x1 + target_width
        
        if x2 > width:
            x2 = width
            x1 = width - target_width
        if x1 < 0:
            x1 = 0
            x2 = target_width
            
        cropped_frame = frame[0:height, x1:x2]
        out.write(cropped_frame)
        
    cap.release()
    out.release()
    
    cmd_merge = [
        "ffmpeg", "-y", "-i", temp_tracked, "-i", temp_clip,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", output_path
    ]
    subprocess.run(cmd_merge, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if os.path.exists(temp_clip): os.remove(temp_clip)
    if os.path.exists(temp_tracked): os.remove(temp_tracked)
    
    return output_path

# ==========================================
# ENDPOINT API UNTUK VERCEL
# ==========================================
@app.post("/api/analyze-youtube")
async def analyze_youtube(req: YTAnalyzeRequest):
    audio_filename = f"outputs/audio_{uuid.uuid4().hex}.m4a"
    try:
        ydl_opts = {
            'format': 'bestaudio/best', 
            'outtmpl': audio_filename, 
            'quiet': True,
            'nocheckcertificate': True,
            'ignoreerrors': True,
            'no_warnings': True
            # TOPENG TV/IOS DIHAPUS BIAR SINKRON SAMA COOKIES WEB
        }
        
        if os.path.exists("cookies.txt"):
            ydl_opts['cookiefile'] = 'cookies.txt'
            
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([req.video_url])
            
        transcript = transcribe_video(audio_filename)
        highlights = analyze_highlights(transcript)
        
        if os.path.exists(audio_filename): os.remove(audio_filename)
        return {"message": "Sukses", "highlights": highlights}
        
    except Exception as e:
        if os.path.exists(audio_filename): os.remove(audio_filename)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/render-youtube-clip")
async def render_youtube_clip(req: YTRenderRequest):
    try:
        output_filename = f"clip_{uuid.uuid4().hex}.mp4"
        output_path = f"outputs/{output_filename}"
        
        ydl_opts = {
            'format': 'best', 
            'quiet': True,
            'nocheckcertificate': True
            # TOPENG TV/IOS DIHAPUS BIAR SINKRON SAMA COOKIES WEB
        }
        
        if os.path.exists("cookies.txt"):
            ydl_opts['cookiefile'] = 'cookies.txt'
            
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(req.video_url, download=False)
            direct_url = info['url']
            
        crop_video_to_short(direct_url, output_path, req.start_seconds, req.end_seconds)
        return {"download_url": f"/outputs/{output_filename}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/outputs/{filename}")
async def download_file(filename: str):
    file_path = os.path.join("outputs", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4", filename=filename)
    raise HTTPException(status_code=404, detail="File tidak ditemukan")
