import whisper
import torch
# from yt_dlp import YoutubeDL

# # YouTube視頻URL
# video_url = "https://www.youtube.com/watch?v=AD4b-52jtos"

# # 設置yt-dlp選項
# ydl_opts = {
#     'format': 'bestaudio/best',
#     'postprocessors': [{
#         'key': 'FFmpegExtractAudio',
#         'preferredcodec': 'mp3',
#         'preferredquality': '192',
#     }],
#     'outtmpl': 'audio.%(ext)s'
# }

# # 下載YouTube視頻音頻
# with YoutubeDL(ydl_opts) as ydl:
#     ydl.download([video_url])

# 使用GPU加載Whisper模型
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device = }")
model = whisper.load_model("large-v3", device=device)

# 流式輸出轉錄結果
def stream_transcribe(audio_file):
    result = model.transcribe(audio_file, fp16=False)
    for segment in result["segments"]:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        print(f"[{start:.2f}s - {end:.2f}s] {text}")

# 調用流式轉錄函數
stream_transcribe("audio.mp3")
