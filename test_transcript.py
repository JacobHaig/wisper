# This file simply runs thorugh json files that are generated from main.py

import time

import json


def load_transcript_from_file(input_path: str) -> dict:
    with open(input_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    return transcript

def print_transcript_in_time(transcript: dict):
    current_time = 0.0

    words = transcript.get('word', [])

    # Sort words by start time just in case
    words.sort(key=lambda x: x.get('start', 0))

    for word_info in words:
        
        start = word_info.get('start', 0)
        end = word_info.get('end', 0)
        word = word_info.get('word', '')

        time.sleep(start - current_time)
        current_time = start

        print(f"{round(start, 2)}s - {round(end, 2)}s : {word}")



def main():
    input_path = "transcript/Triple_kill_at_3_06_59.json"  # Update with your transcript file path
    transcript = load_transcript_from_file(input_path)
    print("Loaded Transcript:")
    # print(json.dumps(transcript, indent=2))

    print_transcript_in_time(transcript)




if __name__ == "__main__":
    main()