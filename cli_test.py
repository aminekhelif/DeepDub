import os
import sys
import argparse
import yaml
import torch
import pathlib
import json
from collections import defaultdict

sys.path.append("../")
from DeepDub.PreProcessing import Preprocessing

def load_config(config_path=None):
    if config_path is None:
        current_file = pathlib.Path(__file__).resolve()  # cli_test.py
        repo_root = current_file.parent
        default_config_path = repo_root / "DeepDub" / "config.yaml"
        config_path = str(default_config_path)

    print(f"Looking for config at: {config_path}")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    else:
        print(f"Config file not found: {config_path}")
        return {}

def get_speaker_count(diar_json_path: str):
    """
    Parse the diarization JSON to find the number of unique speakers.
    Adjust to match your diar.json structure if needed.
    """
    if not os.path.isfile(diar_json_path):
        return None
    with open(diar_json_path, "r") as f:
        diar_data = json.load(f)

    speaker_labels = set()
    if isinstance(diar_data, dict):
        segments = diar_data.get("segments", [])
        for seg in segments:
            label = seg.get("speaker") or seg.get("label")
            if label:
                speaker_labels.add(label)
    elif isinstance(diar_data, list):
        # If diar_data is a list of simplified segments
        for seg in diar_data:
            label = seg.get("speaker") or seg.get("label")
            if label:
                speaker_labels.add(label)

    return len(speaker_labels) if speaker_labels else None


def run_split(preprocessing: Preprocessing):
    print("\n=== Step: Split ===")
    extracted_audio, video_no_audio = preprocessing.split_audio_and_video()
    print(f"Extracted Audio Path: {extracted_audio}")
    print(f"Video Without Audio Path: {video_no_audio}")


def run_separate(preprocessing: Preprocessing):
    print("\n=== Step: Separate ===")
    vocals_path, background_path = preprocessing.separate_audio()
    print(f"Vocals Path: {vocals_path}")
    print(f"Background Path: {background_path}")


def run_diar(preprocessing: Preprocessing):
    print("\n=== Step: Diar ===")
    diar_results = preprocessing.perform_diarization()
    print(f"Diarization Data Path: {diar_results['diarization_data']}")
    print(f"Speaker Audio Dir: {diar_results['speaker_audio_dir']}")
    print(f"Concatenated Audio Dir: {diar_results['concatenated_audio_dir']}")

    # Example of speaker counting
    if os.path.exists(diar_results["diarization_data"]):
        spk_count = get_speaker_count(diar_results["diarization_data"])
        if spk_count is not None:
            print(f"Detected {spk_count} speakers.")
        else:
            print("Could not determine speaker count.")


def process_single_file(
    input_file,
    steps,                       # e.g. ["split","separate","diar"]
    audio_separator_model,
    diarization_batch_size,
    device,
    compute_type,
    HF_token,
    metadata_dir,               # <--- NEW ARG
    num_speakers=None,
    language=None,
    device_index=0
):
    """
    Process a single file using the Preprocessing class with specific steps.
    """
    file_dir = os.path.dirname(input_file)
    file_base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_dir = os.path.join(file_dir, file_base_name, "output_directory")

    # Initialize Preprocessing with the new metadata_dir argument
    preprocessing = Preprocessing(
        input_video=input_file,
        output_dir=output_dir,
        audio_separator_model=audio_separator_model,
        diarization_batch_size=diarization_batch_size,
        device=device,
        compute_type=compute_type,
        HF_token=HF_token,
        num_speakers=num_speakers,
        language=language,
        device_index=device_index,
        metadata_dir=metadata_dir  # pass to Preprocessing
    )

    print(f"\n=== Processing: {input_file} ===")
    print(f"Output dir: {output_dir}")
    print(f"Steps requested: {steps}")

    if "split" in steps:
        run_split(preprocessing)

    if "separate" in steps:
        run_separate(preprocessing)

    if "diar" in steps:
        run_diar(preprocessing)

    # Print final paths
    paths = preprocessing.get_paths()
    print("\nFinal Output Paths:")
    for k, v in paths.items():
        print(f"{k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="Process audio/video files in steps.")
    parser.add_argument("input_path", type=str,
                        help="Path to a single file or directory containing files to process.")
    parser.add_argument("--extensions", type=str, nargs="*",
                        default=None,
                        help="Target extensions to process (mp3, mp4, etc.).")
    parser.add_argument("--audio_separator_model", type=str,
                        default=None,
                        help="Model for audio separation.")
    parser.add_argument("--diarization_batch_size", type=int,
                        default=None,
                        help="Batch size for diarization.")
    parser.add_argument("--compute_type", type=str,
                        default=None,
                        help="Compute precision type for WhisperX (int8, float16, etc.).")
    parser.add_argument("--HF_token", type=str,
                        default=None,
                        help="Hugging Face token if needed.")
    parser.add_argument("--config_file", type=str,
                        default=None,
                        help="Path to config YAML file.")
    parser.add_argument("--steps", type=str,
                        default="split,separate,diar",
                        help="Comma-separated list of steps: split,separate,diar")
    # NEW ARG: metadata_dir
    parser.add_argument("--metadata_dir", type=str,
                        default=None,
                        help="Optional directory where metadata (JSON) is saved. "
                             "If not provided, defaults inside the diarization_dir.")

    args = parser.parse_args()
    config = load_config(args.config_file)

    audio_separator_model = (
        args.audio_separator_model 
        if args.audio_separator_model is not None
        else config.get("audio_separator_model", "Mel-RoFormer")
    )
    diarization_batch_size = (
        args.diarization_batch_size
        if args.diarization_batch_size is not None
        else config.get("diarization_batch_size", 1024)
    )
    compute_type = (
        args.compute_type
        if args.compute_type is not None
        else config.get("compute_type", "int8")
    )
    HF_token = (
        args.HF_token
        if args.HF_token is not None
        else config.get("HF_token", "")
    )

    # Device logic
    if "device" in config and config["device"]:
        device = config["device"]
        device_index = 0
    else:
        if torch.cuda.is_available():
            device = "cuda"
            device_index = 0
        elif torch.backends.mps.is_available():
            device = "mps"
            device_index = 0
        else:
            device = "cpu"
            device_index = 0

    steps_to_run = [s.strip().lower() for s in args.steps.split(",") if s.strip()]

    def is_valid_extension(fname: str, exts) -> bool:
        return fname.lower().endswith(exts)

    # Gather files
    files_to_process = []
    if os.path.isfile(args.input_path):
        files_to_process.append(os.path.abspath(args.input_path))
    else:
        for root, dirs, files in os.walk(args.input_path):
            if "output_directory" in root:
                continue
            for file_name in files:
                if args.extensions:
                    user_exts = tuple("." + e.strip(".").lower() for e in args.extensions)
                    if is_valid_extension(file_name, user_exts):
                        full_path = os.path.join(root, file_name)
                        files_to_process.append(os.path.abspath(full_path))
                else:
                    valid_exts = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac",
                                  ".mp4", ".mkv", ".mov")
                    if is_valid_extension(file_name, valid_exts):
                        full_path = os.path.join(root, file_name)
                        files_to_process.append(os.path.abspath(full_path))

    # Simple logic for english-first if "diar" in steps
    do_english_priority = ("diar" in steps_to_run)
    dir_map = defaultdict(list)
    for fpath in files_to_process:
        parent_dir = os.path.dirname(fpath)
        dir_map[parent_dir].append(fpath)

    for folder_path, file_list in dir_map.items():
        # English first
        english_path = None
        if do_english_priority:
            for f in file_list:
                base = os.path.basename(f).lower()
                if "vo_anglais" in base:
                    english_path = f
                    break

        speaker_count = None
        if english_path and do_english_priority:
            print(f"\n--- [English first] {english_path} ---")
            process_single_file(
                input_file=english_path,
                steps=steps_to_run,
                audio_separator_model=audio_separator_model,
                diarization_batch_size=diarization_batch_size,
                device=device,
                compute_type=compute_type,
                HF_token=HF_token,
                metadata_dir=args.metadata_dir,
            )
            # Optionally parse speaker count
            # Rebuild path to diar.json if needed
            base_name = os.path.splitext(os.path.basename(english_path))[0]
            out_dir = os.path.join(os.path.dirname(english_path), base_name, "output_directory")
            diar_json_path = os.path.join(out_dir, "diarization", "diar.json")
            if os.path.exists(diar_json_path):
                speaker_count = get_speaker_count(diar_json_path)
                print(f"English speaker_count: {speaker_count}")

        # Then process other files
        for f in file_list:
            if f == english_path and do_english_priority:
                continue
            base = os.path.basename(f).lower()
            if do_english_priority and base.startswith("vf"):
                print(f"\n--- [French] {f} ---")
                process_single_file(
                    input_file=f,
                    steps=steps_to_run,
                    audio_separator_model=audio_separator_model,
                    diarization_batch_size=diarization_batch_size,
                    device=device,
                    compute_type=compute_type,
                    HF_token=HF_token,
                    metadata_dir=args.metadata_dir,
                    num_speakers=speaker_count,
                    language="fr",
                )
            else:
                print(f"\n--- [Generic] {f} ---")
                process_single_file(
                    input_file=f,
                    steps=steps_to_run,
                    audio_separator_model=audio_separator_model,
                    diarization_batch_size=diarization_batch_size,
                    device=device,
                    compute_type=compute_type,
                    HF_token=HF_token,
                    metadata_dir=args.metadata_dir
                )


if __name__ == "__main__":
    main()