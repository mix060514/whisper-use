import whisper
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

# 使用Whisper v3轉錄音頻
model = whisper.load_model("large-v3")
result = model.transcribe("audio.mp3")

# 輸出轉錄結果
print(result["text"])
