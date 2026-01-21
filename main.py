import os
import json
import time
import subprocess

import numpy as np
import nemo.collections.asr as asr
from moviepy import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip


def strip_path(path: str) -> str:
    return path.split(".")[0]


def convert_video_to_audio_tracks(video_path, audio_path):
    print("Converting video to all audio tracks with MoviePy...")

    os.makedirs("audio", exist_ok=True)

    print(f"Getting number of audio tracks in {video_path}...")
    ffprobe_cmd = ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "json", video_path]
    result = subprocess.run(ffprobe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    info = json.loads(result.stdout)

    num_tracks = len(info.get("streams", []))
    print(f"Found {num_tracks} audio track(s). Exporting each as mono...")

    audio_paths = []
    for i in range(num_tracks):
        track_output = f"{audio_path}_track{i+1}.mp3"
        # Use ffmpeg to extract each track directly to mono mp3
        extract_cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-map", f"0:a:{i}", "-ac", "1", "-codec:a", "libmp3lame", track_output
        ]
        print(f"Extracting track {i+1} to {track_output}...")
        subprocess.run(extract_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Exported {track_output}")
        audio_paths.append(track_output)

    print("All audio tracks exported.")
    return audio_paths


# This returns a dictionary with word, segment, and char level timestamps
def transcribe_audio_parakeet(audio_paths: list[str]) -> list[dict]:
    print(f"Performing speaker transcription on audio... {audio_paths}")

    asr_model = asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")
    print("type of asr_model:", type(asr_model))
    # type of asr_model: <class 'nemo.collections.asr.models.sortformer_diar_models.SortformerEncLabelModel'>

    outputs = asr_model.transcribe(audio_paths, timestamps=True)
    print(f"Type of output: {type(outputs[0])}")


    output_list = []
    for output in outputs:    
        word_timestamps = output.timestamp['word'] # word level timestamps for first sample
        segment_timestamps = output.timestamp['segment'] # segment level timestamps
        char_timestamps = output.timestamp['char'] # char level timestamps

        if len(segment_timestamps) > 0:
            print("\nSegment-level Timestamps:")
        for stamp in segment_timestamps:
            print(f"  {round(stamp['start'], 2)}s - {round(stamp['end'], 2)}s : {stamp['segment']}")

        output_list.append(output.timestamp)
    return output_list


def save_transcript_to_file(transcript, output_path: str):
    os.makedirs("transcript", exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)
    print(f"Transcript saved to {output_path}")


def video_to_transcript(video_path: str, audio_path: str, transcript_output_path: str):
    audio_paths = convert_video_to_audio_tracks(video_path, audio_path)
    transcript = transcribe_audio_parakeet(audio_paths)

    save_transcript_to_file(transcript, transcript_output_path)


def main():
    for video in os.listdir("video"):
        if video.endswith(('.mp4', '.mov', '.avi', '.mkv')):
            print(f"Processing video: {video}")

            video_path = f"video/{video}"
            audio_path = f"audio/{strip_path(video)}"        
            transcript_output_path = f"transcript/{strip_path(video)}.json"

            print(f"Video Path: {video_path}")
            print(f"Audio Path: {audio_path}")
            print(f"Transcript Output Path: {transcript_output_path}")

            video_to_transcript(video_path, audio_path, transcript_output_path)

            time.sleep(0.5)  # Optional: small delay between processing videos


if __name__ == "__main__":
    main()
