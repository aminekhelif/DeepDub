import os
import json
import torchaudio
import numpy as np
import soundfile as sf
from tqdm import tqdm
import whisperx


class AudioDiarization:
    def __init__(self, audio_path, batch_size=16, device="cpu", compute_type="float16", HF_token=None):
        self.audio_path = os.path.abspath(audio_path)
        self.input_folder = os.path.dirname(self.audio_path)
        self.batch_size = batch_size
        self.device = device
        self.compute_type = compute_type
        self.HF_token = HF_token
        self.diarization_folder = os.path.join(self.input_folder, "diarization")
        self.speaker_audio_folder = os.path.join(self.input_folder, "speaker_audio")
        os.makedirs(self.diarization_folder, exist_ok=True)
        os.makedirs(self.speaker_audio_folder, exist_ok=True)
    
    @staticmethod
    def save_json(data, file_path):
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved: {file_path}")
    
    def perform_diarization(self):
        print("Loading Whisper model...")
        model = whisperx.load_model("large-v3", device=self.device, compute_type=self.compute_type)

        print(f"Loading audio file: {self.audio_path}")
        audio = whisperx.load_audio(self.audio_path)

        print("Transcribing audio...")
        result = model.transcribe(audio, batch_size=self.batch_size)

        print("Loading alignment model...")
        align_model, metadata = whisperx.load_align_model(result["language"], device=self.device)

        print("Aligning transcription...")
        result = whisperx.align(result["segments"], align_model, metadata, audio, self.device)

        print("Performing speaker diarization...")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=self.HF_token, device=self.device)
        diar_segments = diarize_model(audio)

        print("Assigning speaker labels...")
        result = whisperx.assign_word_speakers(diar_segments, result)

        # Save the full diarization result
        full_json_path = os.path.join(self.diarization_folder, "diar.json")
        self.save_json(result, full_json_path)

        # Save simplified diarization without word-level details
        simplified_segments = [
            {k: v for k, v in seg.items() if k != "words"} for seg in result["segments"]
        ]
        simplified_json_path = os.path.join(self.diarization_folder, "diar_simple.json")
        self.save_json(simplified_segments, simplified_json_path)

        print("Diarization complete.")
        return result

    def extract_speaker_audio(self):
        diar_simple_path = os.path.join(self.diarization_folder, "diar_simple.json")
        if not os.path.exists(diar_simple_path):
            raise FileNotFoundError(f"Diarization file not found: {diar_simple_path}")

        with open(diar_simple_path, "r") as f:
            speaker_segments = json.load(f)

        print(f"Loading audio: {self.audio_path}")
        waveform, sample_rate = torchaudio.load(self.audio_path)
        total_audio_duration = waveform.shape[1] / sample_rate

        for idx, segment in enumerate(tqdm(speaker_segments, desc="Processing segments")):
            if any(key not in segment for key in ["start", "end", "text"]):
                print(f"Skipping segment {idx}: Missing required keys.")
                continue

            start, end, speaker = segment["start"], segment["end"], segment.get("speaker", "Unknown")
            if start >= end or start < 0 or end > total_audio_duration:
                print(f"Skipping segment {idx}: Invalid time range.")
                continue

            start_frame, end_frame = int(start * sample_rate), int(end * sample_rate)
            segment_audio = waveform[:, start_frame:end_frame]
            if segment_audio.numel() == 0:
                print(f"Skipping segment {idx}: Empty audio.")
                continue

            speaker_dir = os.path.join(self.speaker_audio_folder, speaker)
            os.makedirs(speaker_dir, exist_ok=True)
            segment_dir = os.path.join(speaker_dir, f"segment_{idx}")
            os.makedirs(segment_dir, exist_ok=True)

            # Save audio and metadata
            audio_path = os.path.join(segment_dir, "audio.wav")
            sf.write(audio_path, segment_audio.numpy().T, sample_rate)
            metadata = {**segment, "audio_file": os.path.relpath(audio_path, self.speaker_audio_folder)}
            self.save_json(metadata, os.path.join(segment_dir, "metadata.json"))

    def get_speakers(self):
        if not os.path.exists(self.speaker_audio_folder):
            raise FileNotFoundError(f"Speaker audio folder not found: {self.speaker_audio_folder}")

        speakers = [
            item for item in os.listdir(self.speaker_audio_folder)
            if os.path.isdir(os.path.join(self.speaker_audio_folder, item))
        ]
        print(f"Found speakers: {speakers}")
        return speakers

    def concatenate_speaker_segments(self):
        for speaker in self.get_speakers():
            speaker_dir = os.path.join(self.speaker_audio_folder, speaker)
            segment_dirs = [
                os.path.join(speaker_dir, d) for d in os.listdir(speaker_dir)
                if os.path.isdir(os.path.join(speaker_dir, d))
            ]
            if not segment_dirs:
                print(f"No segments found for speaker {speaker}.")
                continue

            segments = []
            sample_rates = set()
            for segment_dir in segment_dirs:
                metadata_file = os.path.join(segment_dir, "metadata.json")
                if not os.path.exists(metadata_file):
                    print(f"Skipping segment in {segment_dir}: Metadata not found.")
                    continue
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                audio_file = os.path.join(self.speaker_audio_folder, metadata["audio_file"])
                if not os.path.exists(audio_file):
                    print(f"Skipping segment in {segment_dir}: Audio file not found.")
                    continue

                data, samplerate = sf.read(audio_file)
                sample_rates.add(samplerate)
                segments.append({"data": data, "metadata": metadata})

            if len(sample_rates) > 1:
                print(f"Sample rate mismatch for speaker {speaker}.")
                continue

            sample_rate = sample_rates.pop()
            sorted_segments = sorted(segments, key=lambda x: x["metadata"]["start"])
            concatenated_audio = np.concatenate([s["data"] for s in sorted_segments])
            concatenated_metadata = {"speaker": speaker, "segments": [s["metadata"] for s in sorted_segments]}

            sf.write(os.path.join(speaker_dir, f"{speaker}_concatenated.wav"), concatenated_audio, sample_rate)
            self.save_json(concatenated_metadata, os.path.join(speaker_dir, f"{speaker}_concatenated_metadata.json"))
            print(f"Speaker {speaker} concatenated.")