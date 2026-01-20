import os
from typing import List
from pathlib import Path
from unittest import result
from xml.parsers.expat import model
import json


import torch
from moviepy import VideoFileClip, TextClip, CompositeVideoClip
from nemo.collections.asr.models import SortformerEncLabelModel, ASRModel


from multitalker_transcript_config import MultitalkerTranscriptionConfig
from omegaconf import OmegaConf
from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
from nemo.collections.asr.parts.utils.multispk_transcribe_utils import SpeakerTaggedASR
import soundfile as sf
import numpy as np

# Helper function to strip path and extension
# Usage:
# stripped_name = strip_path("path/to/file.ext")  # returns "file"
def strip_path(path: str) -> str:
    return path.split(".")[0]


def convert_video_to_audio(video_path, audio_path):
    print("Converting video to audio...")

    os.makedirs("audio", exist_ok=True)
    video = VideoFileClip(video_path)

    print(f"Extracting audio to {audio_path}...")
    video.audio.write_audiofile(audio_path)


def transcribe_audio_diarization(audio_path: str):
    print(f"Performing speaker diarization on audio... {audio_path}")

    diar_model: SortformerEncLabelModel = SortformerEncLabelModel.from_pretrained(
        "nvidia/diar_streaming_sortformer_4spk-v2.1")

    print("type of diar_model:", type(diar_model))
    # type of diar_model: <class 'nemo.collections.asr.models.sortformer_diar_models.SortformerEncLabelModel'>

    diar_model.eval()

    diar_model.sortformer_modules.chunk_len = 340
    diar_model.sortformer_modules.chunk_right_context = 40
    diar_model.sortformer_modules.fifo_len = 40
    diar_model.sortformer_modules.spkcache_update_period = 300

    predicted_segments = diar_model.diarize(audio=[audio_path], batch_size=1)

    for segment in predicted_segments:
        print(segment)


def transcribe_audio_multitalker_parakeet(audio_path: str, transcript_output_path: str = "multitalker_transcript.json"):
    print(f"Performing multitalker transcription on audio... {audio_path}")
    print("Torch device:", torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Preprocess audio file to ensure mono (1D) shape ---
    data, samplerate = sf.read(audio_path)
    # If stereo or multi-channel, convert to mono by averaging channels
    if len(data.shape) > 1:
        print(f"Audio has {data.shape[1]} channels, converting to mono.")
        data = np.mean(data, axis=1)
        sf.write(audio_path, data, samplerate)
    elif len(data.shape) == 1:
        print("Audio is already mono.")
    else:
        raise RuntimeError(
            "Unexpected audio data shape: {}".format(data.shape))

    # --- Continue with NeMo pipeline ---
    diar_model = SortformerEncLabelModel.from_pretrained("nvidia/diar_streaming_sortformer_4spk-v2").eval().to(device)
    asr_model = ASRModel.from_pretrained("nvidia/multitalker-parakeet-streaming-0.6b-v1").eval().to(device)


    # Set up the configuration for multitalker transcription
    cfg = OmegaConf.structured(MultitalkerTranscriptionConfig())
    cfg.audio_file = audio_path
    cfg.output_path = transcript_output_path
    diar_model = MultitalkerTranscriptionConfig.init_diar_model( cfg, diar_model)

    samples = [{'audio_filepath': cfg.audio_file}]
    streaming_buffer = CacheAwareStreamingAudioBuffer(
        model=asr_model,
        online_normalization=cfg.online_normalization,
        pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
    )
    streaming_buffer.append_audio_file(audio_filepath=cfg.audio_file, stream_id=-1)
    streaming_buffer_iter = iter(streaming_buffer)

    # Use the helper class `SpeakerTaggedASR`, which handles all ASR and diarization cache data for streaming.
    multispk_asr_streamer = SpeakerTaggedASR(cfg, asr_model, diar_model)

    # Iterate over audio chunks and perform streaming ASR with speaker tagging.
    for step_num, (chunk_audio, chunk_lengths) in enumerate(streaming_buffer_iter):
        drop_extra_pre_encoded = ( 0 if step_num == 0 and not cfg.pad_and_drop_preencoded
            else asr_model.encoder.streaming_cfg.drop_extra_pre_encoded
        )
        with torch.inference_mode():
            with torch.amp.autocast(diar_model.device.type, enabled=True):
                with torch.no_grad():
                    multispk_asr_streamer.perform_parallel_streaming_stt_spk(
                        step_num=step_num,
                        chunk_audio=chunk_audio,
                        chunk_lengths=chunk_lengths,
                        is_buffer_empty=streaming_buffer.is_buffer_empty(),
                        drop_extra_pre_encoded=drop_extra_pre_encoded,
                    )

    # Generate the speaker-tagged transcript and print it.
    multispk_asr_streamer.generate_seglst_dicts_from_parallel_streaming(samples=samples)
    transcript = multispk_asr_streamer.instance_manager.seglst_dict_list

    return transcript


def save_transcript_to_file(transcript, output_path: str):
    
    os.makedirs("transcript", exist_ok=True)

    import torch
    def tensor_to_list(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: tensor_to_list(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [tensor_to_list(v) for v in obj]
        return obj

    serializable_transcript = tensor_to_list(transcript)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_transcript, f, ensure_ascii=False, indent=2)
    print(f"Transcript saved to {output_path}")


def video_to_transcript(video_path: str, audio_path: str, transcript_output_path: str):
    convert_video_to_audio(video_path, audio_path)
    transcript = transcribe_audio_multitalker_parakeet(audio_path)

    save_transcript_to_file(transcript, transcript_output_path)


def main():
    for video in os.listdir("video"):
        if video.endswith(('.mp4', '.mov', '.avi', '.mkv')):
            print(f"Processing video: {video}")

            video_path = f"video/{video}"
            audio_path = f"audio/{strip_path(video)}.mp3"
            transcript_output_path = f"transcript/{strip_path(video)}.json"

            print(f"Video Path: {video_path}")
            print(f"Audio Path: {audio_path}")
            print(f"Transcript Output Path: {transcript_output_path}")

            video_to_transcript(video_path, audio_path, transcript_output_path)


if __name__ == "__main__":
    main()
