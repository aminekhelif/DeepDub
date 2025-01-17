import os
import sys
import argparse
import yaml
import torch
import pathlib
import json
from collections import defaultdict

# Adjust your imports according to your project
sys.path.append("../")
from DeepDub.PreProcessing import Preprocessing

###############################################################################
# 1) Minimal Helpers for Metadata
###############################################################################
def load_metadata(meta_path: str) -> dict:
    """
    Loads the metadata JSON if it exists, otherwise returns an empty dict.
    Structure (example):
    {
      "files": {
        "/full/path/to/input.mp4": {
           "split_audio": "/full/path/to/input_audio.mp3",
           "video_no_audio": "/full/path/to/input_no_audio.mp4",
           "vocals": "/full/path/to/vocals.mp3",
           "background": "/full/path/to/instrumental.mp3",
           "diarization_data": "/full/path/to/diar_simple.json"
           # ...
        },
        ...
      }
    }
    """
    if os.path.isfile(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    else:
        return {"files": {}}
    
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

def save_metadata(meta: dict, meta_path: str):
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)

def update_metadata_for_file(
    meta: dict, input_file: str, 
    split_audio=None, video_no_audio=None,
    vocals=None, background=None,
    diarization_data=None
):
    """
    Updates the metadata dictionary for a given input file.
    We only update the keys that are given as non-None.
    """
    if "files" not in meta:
        meta["files"] = {}

    if input_file not in meta["files"]:
        meta["files"][input_file] = {}

    if split_audio is not None:
        meta["files"][input_file]["split_audio"] = split_audio
    if video_no_audio is not None:
        meta["files"][input_file]["video_no_audio"] = video_no_audio
    if vocals is not None:
        meta["files"][input_file]["vocals"] = vocals
    if background is not None:
        meta["files"][input_file]["background"] = background
    if diarization_data is not None:
        meta["files"][input_file]["diarization_data"] = diarization_data

###############################################################################
# 2) The Core Steps: Split, Separate, Diar
###############################################################################
def run_split(preprocessing: Preprocessing, meta: dict):
    """
    Runs the 'split_audio_and_video' step.
    Updates the metadata with the resulting audio + video_no_audio.
    """
    print("\n=== Step: Split ===")
    extracted_audio, video_no_audio = preprocessing.split_audio_and_video()
    print(f"Extracted Audio Path: {extracted_audio}")
    print(f"Video Without Audio Path: {video_no_audio}")

    # Update metadata
    input_file = preprocessing.input_video
    update_metadata_for_file(
        meta, input_file, 
        split_audio=extracted_audio, 
        video_no_audio=video_no_audio
    )

def run_separate(preprocessing: Preprocessing, meta: dict):
    """
    Runs the 'separate_audio' step.
    Updates the metadata with the resulting vocals + background.
    """
    print("\n=== Step: Separate ===")
    vocals, background = preprocessing.separate_audio()
    print(f"Vocals Path: {vocals}")
    print(f"Background Path: {background}")

    # Update metadata
    input_file = preprocessing.input_video
    update_metadata_for_file(
        meta, input_file,
        vocals=vocals,
        background=background
    )

def run_diar(preprocessing: Preprocessing, meta: dict):
    """
    Runs the 'perform_diarization' step.
    Updates the metadata with path to the diarization data (e.g. diar_simple.json).
    """
    print("\n=== Step: Diar ===")
    diarization_results = preprocessing.perform_diarization()
    diar_data = diarization_results["diarization_data"]
    print(f"Diarization Data: {diar_data}")
    print(f"Speaker Audio Directory: {diarization_results['speaker_audio_dir']}")
    print(f"Concatenated Audio Directory: {diarization_results['concatenated_audio_dir']}")

    # Update metadata
    input_file = preprocessing.input_video
    update_metadata_for_file(
        meta, input_file,
        diarization_data=diar_data
    )

###############################################################################
# 3) Additional Helper: Speaker Count
###############################################################################
def get_speaker_count(diar_json_path: str):
    """
    Attempts to parse the diarization JSON file to count unique speakers.
    Adapt the parsing to match your actual diar.json or diar_simple.json structure.
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
        for seg in diar_data:
            label = seg.get("speaker") or seg.get("label")
            if label:
                speaker_labels.add(label)

    return len(speaker_labels) if speaker_labels else None

###############################################################################
# 4) process_single_file logic
###############################################################################
def process_single_file(
    input_file: str,
    steps: list,
    audio_separator_model: str,
    diarization_batch_size: int,
    device: str,
    compute_type: str,
    HF_token: str,
    meta: dict,
    num_speakers=None,
    language=None,
    device_index=0
):
    """
    Given an input_file and steps, process it using Preprocessing.
    We also pass 'meta' so we can see if the user already did split, etc.
    """
    file_dir = os.path.dirname(input_file)
    file_base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_dir = os.path.join(file_dir, file_base_name, "output_directory")

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
        device_index=device_index
    )

    # 1) If "split" is in steps, we do it - else we check if it was already done (via meta).
    do_split = "split" in steps
    do_separate = "separate" in steps
    do_diar = "diar" in steps

    # "SPLIT" step
    if do_split:
        # We run it only if not done previously or user explicitly wants to run again
        run_split(preprocessing, meta)
    else:
        # If user is *skipping* split, we must ensure the Preprocessing object
        # has the "self.extracted_audio_path" set to whatever was done previously
        # This is so that 'separate_audio()' can find the correct input path.
        prev_audio = meta["files"].get(input_file, {}).get("split_audio")
        if prev_audio and os.path.isfile(prev_audio):
            preprocessing.extracted_audio_path = prev_audio
            # We do NOT run 'split_audio_and_video()' again
        else:
            # Possibly raise a warning or handle the fact that there's no splitted file
            print(f"Warning: No existing split audio found for {input_file}, skipping 'split' step.")
            return  # or you can decide to forcibly re-split

    # "SEPARATE" step
    if do_separate:
        # If the user wants to separate, we call run_separate
        run_separate(preprocessing, meta)
    else:
        # If user is skipping separate, we must load the previously saved "vocals" path
        prev_vocals = meta["files"].get(input_file, {}).get("vocals")
        if prev_vocals and os.path.isfile(prev_vocals):
            preprocessing.vocals_path = prev_vocals
        else:
            print(f"Warning: No existing 'vocals' file found for {input_file}, skipping 'separate' step.")
            return

    # "DIAR" step
    if do_diar:
        # We do the English-over-French logic *only here* if you want:
        run_diar(preprocessing, meta)
    else:
        # If skipping diar, we might not do anything
        pass

    # Print final paths
    print("\nFinal Output Paths:")
    paths = preprocessing.get_paths()
    for k, v in paths.items():
        print(f"{k}: {v}")


###############################################################################
# 5) Main
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Process audio/video files in separate steps.")
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
        help="One or more file extensions to process (e.g. mp3, mp4)."
    )
    parser.add_argument(
        "--audio_separator_model",
        type=str,
        default=None,
        help="Which model for audio separation."
    )
    parser.add_argument(
        "--diarization_batch_size",
        type=int,
        default=None,
        help="Batch size for diarization."
    )
    parser.add_argument(
        "--compute_type",
        type=str,
        default=None,
        help="WhisperX compute type: 'int8', 'float16', etc."
    )
    parser.add_argument(
        "--HF_token",
        type=str,
        default=None,
        help="Hugging Face token if needed."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to a config YAML file."
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="metadata.json",
        help="Where to store/read the JSON metadata about splitted/separated/diarized files."
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="split,separate,diar",
        help="Comma-separated list of steps to run: [split, separate, diar]."
    )

    args = parser.parse_args()

    # Load YAML config
    config = load_config(args.config_file)

    # Merge CLI -> config defaults
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

    # Steps
    steps_to_run = [s.strip().lower() for s in args.steps.split(",") if s.strip()]

    # Build extension list
    if args.extensions:
        user_extensions = tuple("." + ext.strip(".").lower() for ext in args.extensions)
    else:
        # fallback
        user_extensions = None

    # Load existing metadata or create new
    meta_path = args.metadata_file
    meta = load_metadata(meta_path)

    # Gather files
    input_path = args.input_path
    files_to_process = []

    def is_valid_extension(fname: str, exts) -> bool:
        return fname.lower().endswith(exts)

    if os.path.isfile(input_path):
        files_to_process.append(input_path)
    else:
        for root, dirs, files in os.walk(input_path):
            if "output_directory" in root:
                continue
            for f in files:
                if user_extensions:
                    if is_valid_extension(f, user_extensions):
                        full_path = os.path.join(root, f)
                        files_to_process.append(full_path)
                else:
                    valid_exts = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac",
                                  ".mp4", ".mkv", ".mov")
                    if is_valid_extension(f, valid_exts):
                        full_path = os.path.join(root, f)
                        files_to_process.append(full_path)

    # Now, if the user only wants to do 'split' or 'separate', we can skip the English-first logic
    # We only do the English-over-French priority if 'diar' is in steps.
    if "diar" in steps_to_run:
        # We'll do your English -> French logic only for the diar step
        dir_map = defaultdict(list)
        for fpath in files_to_process:
            dir_map[os.path.dirname(fpath)].append(fpath)

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
                print(f"\n--- Processing English file (for diar): {english_path} ---")
                process_single_file(
                    input_file=english_path,
                    steps=steps_to_run,
                    audio_separator_model=audio_separator_model,
                    diarization_batch_size=diarization_batch_size,
                    device=device,
                    compute_type=compute_type,
                    HF_token=HF_token,
                    meta=meta,
                    num_speakers=None,
                    language="en",
                    device_index=device_index
                )
                # If diar was actually run, get speaker count
                if "diar" in steps_to_run:
                    # retrieve diar.json from meta
                    diar_json = meta["files"].get(english_path, {}).get("diarization_data")
                    if diar_json:
                        speaker_count = get_speaker_count(diar_json)
                        print(f"English speaker_count: {speaker_count}")

            # Process French or other
            for fpath in file_list:
                if fpath == english_path:
                    continue
                base = os.path.basename(fpath).lower()
                if base.startswith("vf"):
                    print(f"\n--- Processing French file (for diar): {fpath} ---")
                    process_single_file(
                        input_file=fpath,
                        steps=steps_to_run,
                        audio_separator_model=audio_separator_model,
                        diarization_batch_size=diarization_batch_size,
                        device=device,
                        compute_type=compute_type,
                        HF_token=HF_token,
                        meta=meta,
                        num_speakers=speaker_count,
                        language="fr",
                        device_index=device_index
                    )
                else:
                    print(f"\n--- Processing other file (for diar): {fpath} ---")
                    process_single_file(
                        input_file=fpath,
                        steps=steps_to_run,
                        audio_separator_model=audio_separator_model,
                        diarization_batch_size=diarization_batch_size,
                        device=device,
                        compute_type=compute_type,
                        HF_token=HF_token,
                        meta=meta,
                        num_speakers=None,
                        language=None,
                        device_index=device_index
                    )
    else:
        # If we do NOT have 'diar' in steps, we can just process all files normally
        for fpath in files_to_process:
            process_single_file(
                input_file=fpath,
                steps=steps_to_run,
                audio_separator_model=audio_separator_model,
                diarization_batch_size=diarization_batch_size,
                device=device,
                compute_type=compute_type,
                HF_token=HF_token,
                meta=meta,
                num_speakers=None,   # Not relevant if not running diar
                language=None,
                device_index=device_index
            )

    # Finally, save the updated metadata
    save_metadata(meta, meta_path)
    print(f"\n=== All done! Updated metadata saved to: {meta_path} ===")


if __name__ == "__main__":
    main()