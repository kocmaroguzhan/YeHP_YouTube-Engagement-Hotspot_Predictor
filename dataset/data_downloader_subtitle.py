import yt_dlp
import os
import json
from concurrent.futures import ThreadPoolExecutor

video_urls = [
    "https://www.youtube.com/watch?v=pZz3tfXEFmU&ab_channel=Vox"
]

def process_video(url):
    video_id = "unknown_id"  # Fallback if metadata extraction fails
    folder_name = "unknown_video"
    
    try:
        # Step 1: Extract metadata
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)

        video_id = info['id']
        folder_name = video_id
        os.makedirs(folder_name, exist_ok=True)

        # Step 2: Save metadata
        metadata_path = os.path.join(folder_name, f"{video_id}_metadata.json")
        with open(metadata_path, "w", encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=4)

        # Output template
        output_template = os.path.join(folder_name, f"{video_id}_%(suffix)s.%(ext)s")

        # Step 3: Download subtitles
        subtitles_opts = {
            'outtmpl': output_template.replace("%(suffix)s", "sub"),
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'skip_download': True,
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(subtitles_opts) as ydl:
            ydl.download([url])

        print(f"✅ Finished: {video_id}")

    except Exception as e:
        os.makedirs(folder_name, exist_ok=True)  # Ensure folder exists even if metadata failed
        error_path = os.path.join(folder_name, "error.log")
        with open(error_path, "a", encoding='utf-8') as err_file:
            err_file.write(f"❌ Error processing {url}:\n{str(e)}\n\n")
        print(f"❌ Failed: {url} — Error logged in {error_path}")

# Parallel execution
with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(process_video, video_urls)
