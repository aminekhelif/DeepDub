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
        # Build an absolute path relative to THIS file (cli_test.py),
        current_file = pathlib.Path(__file__).resolve()  # cli_test.py
        repo_root = current_file.parent           
        default_config_path = repo_root / "DeepDub" / "config.yaml"
        config_path = str(default_config_path)

    print(f"Looking for config at: {config_path}")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    else:
        print(f"Config file not found. {config_path}")
        return {}

def process_single_file(
    input_file,
    audio_separator_model,
    diarization_batch_size,
    device,
    compute_type,
    HF_token,
    num_speakers=None,
    language=None,
    device_index=0
):
    """
    Process a single file (audio or video) using the Preprocessing class.
    Returns the diarization_results dict so we can parse #speakers if needed.
    """
    # Create an output directory next to the input file
    file_dir = os.path.dirname(input_file)
    file_base_name = os.path.splitext(os.path.basename(input_file))[0]  # e.g. "my_audio" (no extension)
    output_dir = os.path.join(file_dir, file_base_name, "output_directory")

    # Initialize the Preprocessing class
    preprocessing = Preprocessing(
        input_video=input_file,
        output_dir=output_dir,
        audio_separator_model=audio_separator_model,
        diarization_batch_size=diarization_batch_size,
        device=device,
        device_index=device_index,
        compute_type=compute_type,
        HF_token=HF_token,
        num_speakers=num_speakers,
        language=language
    )

    print(f"\n=== Processing: {input_file} ===")
    print(f"Output directory: {output_dir}")

    # Step 1: Split the file into audio & video-without-audio
    print("\nStep 1: Splitting Audio and Video")
    extracted_audio, video_no_audio = preprocessing.split_audio_and_video()
    print(f"Extracted Audio Path: {extracted_audio}")
    print(f"Video Without Audio Path: {video_no_audio}")

    # Step 2: Separate the audio
    print("\nStep 2: Separating Audio")
    vocals, background = preprocessing.separate_audio()
    print(f"Vocals Path: {vocals}")
    print(f"Background Path: {background}")

    # Step 3: Diarization
    print("\nStep 3: Performing Diarization")
    diarization_results = preprocessing.perform_diarization()
    print("Diarization Results:")
    print(f"Diarization Data Path: {diarization_results['diarization_data']}")
    print(f"Speaker Audio Directory: {diarization_results['speaker_audio_dir']}")
    print(f"Concatenated Audio Directory: {diarization_results['concatenated_audio_dir']}")

    print("\nFinal Output Paths:")
    paths = preprocessing.get_paths()
    for key, value in paths.items():
        print(f"{key}: {value}")

    # Return the diarization results so we can parse #speakers outside
    return diarization_results


def get_speaker_count(diar_json_path):
    """
    Given a diarization JSON file path, return
    the number of unique speakers found.
    """
    if not os.path.isfile(diar_json_path):
        return None

    with open(diar_json_path, "r") as f:
        diar_data = json.load(f)

    # Adjust if your diar.json structure is different;
    # below we assume a list of segments with "label": "SPEAKER_XX".
    speaker_labels = set()
    for seg in diar_data.get("segments", []):
        label = seg.get("label")
        if label is not None:
            speaker_labels.add(label)

    return len(speaker_labels) if speaker_labels else None


def main():
    """
    Main function to parse arguments and process either:
    - A single file (audio or video)
    - Or crawl a directory for files with specified target extensions
    """
    parser = argparse.ArgumentParser(description="Process audio/video files.")
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to a single file or directory containing files to process."
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="*",
        default=None,
        help="One or more target extensions to process (e.g. --extensions mp3 mp4). "
             "If omitted and input_path is a directory, only certain default A/V extensions are processed."
    )
    # For the remaining arguments, default=None so we can detect if the user set them or not.
    parser.add_argument(
        "--audio_separator_model",
        type=str,
        default=None,
        help="Model to use for audio separation (overrides config if provided)."
    )
    parser.add_argument(
        "--diarization_batch_size",
        type=int,
        default=None,
        help="Batch size for diarization steps (overrides config if provided)."
    )
    parser.add_argument(
        "--compute_type",
        type=str,
        default=None,
        help="Compute type for WhisperX inference (e.g., 'int8', 'float16', 'float32')."
    )
    parser.add_argument(
        "--HF_token",
        type=str,
        default=None,
        help="Optional Hugging Face token if needed (overrides config if provided)."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to your config YAML file. Defaults to 'config.yaml'."
    )

    args = parser.parse_args()

    # 1. Load the YAML config file.
    config = load_config(args.config_file)

    # 2. If an argument is provided on the CLI, use that. Otherwise fallback to config, then fallback to a hard-coded default.

    # - Audio separator model
    audio_separator_model = (
        args.audio_separator_model 
        if args.audio_separator_model is not None 
        else config.get("audio_separator_model", "Mel-RoFormer")  # fallback
    )

    # - Diarization batch size
    diarization_batch_size = (
        args.diarization_batch_size
        if args.diarization_batch_size is not None
        else config.get("diarization_batch_size", 1024)  # fallback
    )

    # - Compute type
    compute_type = (
        args.compute_type
        if args.compute_type is not None
        else config.get("compute_type", "int8")  # fallback
    )

    # - HF Token
    HF_token = (
        args.HF_token 
        if args.HF_token is not None
        else config.get("HF_token", "")  # fallback
    )

    # - Device logic
    #   If `device` is in the config, we use it. Otherwise, we detect automatically.
    if "device" in config and config["device"]:
        device = config["device"]
    else:
        # Automatic detection
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                device_index = list(range(num_gpus))
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
                device_index = 0
            else:
                device = 'cpu'
                device_index = 0

    def is_valid_extension(fname: str, exts) -> bool:
        """Helper to check if a file has an allowed extension."""
        return fname.lower().endswith(exts)

    # If user provided extensions, build a tuple of them
    if args.extensions:
        user_extensions = tuple("." + ext.strip(".").lower() for ext in args.extensions)
    else:
        user_extensions = None

    # ------------------------------------------------------
    #  Pre-snapshot Implementation: Gather all files first,
    #  skip "output_directory" if found, then group by folder
    # ------------------------------------------------------
    files_to_process = []

    # If input_path is a single file, just add it
    if os.path.isfile(args.input_path):
        files_to_process.append(args.input_path)
    else:
        for root, dirs, files in os.walk(args.input_path):
            # Skip any directory path that includes 'output_directory'
            if "output_directory" in root:
                continue

            for file_name in files:
                if user_extensions:
                    # If user gave custom extensions
                    if is_valid_extension(file_name, user_extensions):
                        full_path = os.path.join(root, file_name)
                        files_to_process.append(full_path)
                else:
                    # Fallback to default A/V extensions
                    valid_exts = (".mp3", ".wav", ".mp4", ".mkv", ".mov")
                    if is_valid_extension(file_name, valid_exts):
                        full_path = os.path.join(root, file_name)
                        files_to_process.append(full_path)

    # Group files by parent directory
    dir_map = defaultdict(list)
    for fpath in files_to_process:
        parent_dir = os.path.dirname(fpath)
        dir_map[parent_dir].append(fpath)

    # Process English (VO_anglais) first to get speaker count, then French (VF*, VFF*, VFQ*) with that count
    for folder_path, file_list in dir_map.items():
        english_path = None
        for fpath in file_list:
            base = os.path.basename(fpath).lower()
            if "vo_anglais" in base:
                english_path = fpath
                break

        speaker_count = None
        # Process English first
        if english_path:
            diar_results = process_single_file(
                input_file=english_path,
                audio_separator_model=audio_separator_model,
                diarization_batch_size=diarization_batch_size,
                device=device,
                compute_type=compute_type,
                HF_token=HF_token,
                language="en",
                num_speakers=None,
                device_index=device_index
            )
            from_json = diar_results.get("diarization_data")
            speaker_count = get_speaker_count(from_json)
            print(f"English speaker_count: {speaker_count}")

        # Now process French files in the same folder
        for fpath in file_list:
            if fpath == english_path:
                continue
            base = os.path.basename(fpath).lower()
            # If it starts with 'vf' or 'vff' or 'vfq'...
            if base.startswith("vf"):
                process_single_file(
                    input_file=fpath,
                    audio_separator_model=audio_separator_model,
                    diarization_batch_size=diarization_batch_size,
                    device=device,
                    compute_type=compute_type,
                    HF_token=HF_token,
                    language="fr",
                    num_speakers=speaker_count,
                    device_index=device_index
                )
            else:
                # For now, skip all else
                pass


if __name__ == "__main__":
    main()