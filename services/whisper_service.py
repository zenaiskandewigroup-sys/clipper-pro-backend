import os
import subprocess
from groq import Groq

def transcribe_video(video_path: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise Exception("API Key Groq belum dipasang di sistem!")
        
    client = Groq(api_key=api_key)
    audio_path = video_path + ".mp3"
    
    # 1. Ekstrak audio dari video (Groq butuh file audio, maks 25MB)
    # Kompres jadi 32k biar file 1 jam pun ukurannya super kecil
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path, 
        "-vn", "-c:a", "libmp3lame", "-b:a", "32k", audio_path
    ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # 2. Kirim ke otak Groq
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), audio_file.read()),
            model="whisper-large-v3",
            response_format="verbose_json",
        )
        
    # Bersihkan file audio sampah
    if os.path.exists(audio_path):
        os.remove(audio_path)
        
    # 3. Format hasil transkrip
    formatted_transcript = []
    
    # Antisipasi perbedaan output dict/object pada versi SDK Groq
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
