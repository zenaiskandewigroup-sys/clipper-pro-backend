import os
import json
from google import genai
from google.genai import types

def analyze_highlights(transcript_text: str):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise Exception("API Key Gemini belum dipasang di file .env!")
        
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    Kamu adalah Editor Video TikTok Profesional.
    Analisis transkrip ini dan pilih 1-3 momen viral (durasi ideal 30-60 detik).
    Return HANYA format array JSON valid persis seperti ini, tanpa teks tambahan apapun:
    [
      {{
        "title": "Judul Menarik",
        "start_seconds": 15,
        "end_seconds": 60,
        "reason": "Alasan klip ini viral",
        "hook_text": "Kutipan hook pertama",
        "suggested_titles": ["Opsi Judul 1", "Opsi Judul 2", "Opsi Judul 3"]
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
    
    # --- FITUR PEMBERSIH MARKDOWN ---
    raw_json = response.text.strip()
    if raw_json.startswith("```json"):
        raw_json = raw_json[7:]
    elif raw_json.startswith("```"):
        raw_json = raw_json[3:]
        
    if raw_json.endswith("```"):
        raw_json = raw_json[:-3]
        
    try:
        return json.loads(raw_json.strip())
    except json.JSONDecodeError:
        print(f"❌ TEKS GAGAL DIPARSING: \n{response.text}")
        raise Exception("Gagal membaca respons dari Gemini karena format tidak sesuai.")
