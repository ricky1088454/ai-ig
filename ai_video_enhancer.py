import os
import yt_dlp
import cv2
import torch
from django.http import JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from realesrgan import RealESRGAN
from django.shortcuts import render
import ffmpeg
import threading
from tqdm import tqdm

# Setup Directories
DOWNLOAD_FOLDER = os.path.join(settings.BASE_DIR, "downloads")
PROCESSED_FOLDER = os.path.join(settings.BASE_DIR, "processed")
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# YouTube Downloader with Progress

def download_video(url):
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
        'progress_hooks': [progress_hook],
        'outtmpl': f'{DOWNLOAD_FOLDER}/%(title)s.%(ext)s'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        return os.path.join(DOWNLOAD_FOLDER, f"{info_dict['title']}.{info_dict['ext']}")

def progress_hook(d):
    if d['status'] == 'downloading':
        print(f"Downloading: {d['_percent_str']} {d['_eta_str']}")

# Video Enhancement with AI & Multi-threading
def enhance_video(input_path, output_path):
    model = RealESRGAN(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_weights("weights/RealESRGAN_x4plus.pth")
    
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(3))*4, int(cap.get(4))*4))
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=frame_count, desc="Enhancing Video", unit="frame")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        enhanced_frame = model.enhance(frame)
        out.write(enhanced_frame)
        progress_bar.update(1)
    
    cap.release()
    out.release()
    progress_bar.close()
    return output_path

# Video Audio Enhancement using FFmpeg
def enhance_audio(input_path, output_path):
    ffmpeg.input(input_path).output(output_path, af='highpass=200, lowpass=3000').run()
    return output_path

# Video Processing Pipeline
def process_pipeline(video_url):
    video_path = download_video(video_url)
    enhanced_video_path = os.path.join(PROCESSED_FOLDER, os.path.basename(video_path))
    
    threading.Thread(target=enhance_video, args=(video_path, enhanced_video_path)).start()
    audio_output = enhanced_video_path.replace(".mp4", "_audio.mp4")
    threading.Thread(target=enhance_audio, args=(video_path, audio_output)).start()
    
    return enhanced_video_path

@csrf_exempt
def process_video(request):
    if request.method == "POST":
        data = request.POST
        video_url = data.get("url")
        
        if not video_url:
            return JsonResponse({"error": "No URL provided"}, status=400)

        processed_video = process_pipeline(video_url)
        return FileResponse(open(processed_video, "rb"), as_attachment=True)

# Django URL Routing
from django.urls import path
urlpatterns = [
    path('process/', process_video, name='process_video'),
]
