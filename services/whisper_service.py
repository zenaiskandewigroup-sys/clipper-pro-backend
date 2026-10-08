from faster_whisper import WhisperModel

# Pakai model base dan compute_type int8 biar enteng di CPU HP
model = WhisperModel("tiny", device="cpu", compute_type="int8")

def transcribe_video(video_path: str) -> str:
    segments, _ = model.transcribe(video_path, beam_size=5)
    formatted_transcript = []
    for segment in segments:
        start_min = int(segment.start // 60)
        start_sec = int(segment.start % 60)
        end_min = int(segment.end // 60)
        end_sec = int(segment.end % 60)
        timestamp = f"[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}]"
        formatted_transcript.append(f"{timestamp} {segment.text.strip()}")
    return "\n".join(formatted_transcript)
