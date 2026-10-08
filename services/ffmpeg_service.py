import os
import subprocess
import cv2

def crop_video_to_short(input_path: str, output_path: str, start_sec: int, end_sec: int):
    duration = end_sec - start_sec
    temp_clip = "temp_raw_clip.mp4"
    
    # 1. Potong klip kasar (Memudahkan OpenCV jalan lebih cepat)
    cmd_cut = [
        "ffmpeg", "-y", "-ss", str(start_sec), "-i", input_path,
        "-t", str(duration), "-c:v", "copy", "-c:a", "copy", temp_clip
    ]
    subprocess.run(cmd_cut, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # 2. Proses Face Tracking OpenCV per Frame
    cap = cv2.VideoCapture(temp_clip)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps: fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    target_width = int(height * 9 / 16) # Format 9:16 Vertikal
    
    temp_tracked = "temp_tracked.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_tracked, fourcc, fps, (target_width, height))
    
    # Load model pendeteksi wajah bawaan OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    last_center_x = width // 2
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Deteksi wajah
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            # Ambil wajah paling besar
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            x, y, w, h = faces[0]
            current_center_x = x + w // 2
            # Smoothing pergerakan kamera biar gak kaku (10% pergerakan baru, 90% posisi lama)
            last_center_x = int(last_center_x * 0.9 + current_center_x * 0.1)
            
        # Kalkulasi crop 9:16
        x1 = max(0, last_center_x - target_width // 2)
        x2 = x1 + target_width
        
        # Mencegah keluar batas frame
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
    
    # 3. Gabungkan video yg sudah dilacak wajahnya dengan Audio aslinya
    # Menggunakan preset ultrafast biar CPU HP gak keberatan
    cmd_merge = [
        "ffmpeg", "-y",
        "-i", temp_tracked,
        "-i", temp_clip,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        output_path
    ]
    subprocess.run(cmd_merge, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Hapus file sampah sementara
    if os.path.exists(temp_clip): os.remove(temp_clip)
    if os.path.exists(temp_tracked): os.remove(temp_tracked)
    
    return output_path
