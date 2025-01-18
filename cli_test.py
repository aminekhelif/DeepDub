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

###############################################################################
# 1) Minimal Helpers for Metadata
###############################################################################
def load_metadata(meta_path: str) -> dict:
    if os.path.isfile(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    else:
        return {"files": {}}

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

def save_metadata(meta: dict, meta_path: str):
    dir_path = os.path.dirname(meta_path) or "."
    os.makedirs(dir_path, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)

def update_metadata_for_file(
    meta: dict, input_file: str, 
    split_audio=None, video_no_audio=None,
    vocals=None, background=None,
    diarization_data=None
):
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
def run_split(preprocessing: Preprocessing, meta: dict, meta_path: str):
    print("\n=== Step: Split ===")
    extracted_audio, video_no_audio = preprocessing.split_audio_and_video()
    print(f"Extracted Audio Path: {extracted_audio}")
    print(f"Video Without Audio Path: {video_no_audio}")

    input_file = preprocessing.input_video
    update_metadata_for_file(
        meta, input_file, 
        split_audio=extracted_audio, 
        video_no_audio=video_no_audio
    )
    save_metadata(meta, meta_path)

def run_separate(preprocessing: Preprocessing, meta: dict, meta_path: str):
    print("\n=== Step: Separate ===")
    vocals, background = preprocessing.separate_audio()
    print(f"Vocals Path: {vocals}")
    print(f"Background Path: {background}")

    input_file = preprocessing.input_video
    update_metadata_for_file(
        meta, input_file,
        vocals=vocals,
        background=background
    )
    save_metadata(meta, meta_path)

def run_diar(preprocessing: Preprocessing, meta: dict, meta_path: str):
    print("\n=== Step: Diar ===")
    diarization_results = preprocessing.perform_diarization()
    diar_data = diarization_results["diarization_data"]
    print(f"Diarization Data: {diar_data}")
    print(f"Speaker Audio Directory: {diarization_results['speaker_audio_dir']}")
    print(f"Concatenated Audio Directory: {diarization_results['concatenated_audio_dir']}")

    input_file = preprocessing.input_video
    update_metadata_for_file(
        meta, input_file,
        diarization_data=diar_data
    )
    save_metadata(meta, meta_path)

###############################################################################
# 3) Additional Helper: Speaker Count
###############################################################################
def get_speaker_count(diar_json_path: str):
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
# 4) Step Skip Checks (NEW)
###############################################################################
def split_already_done(meta: dict, input_file: str) -> bool:
    """
    Returns True if 'split_audio' is in the metadata for this file
    AND the file physically exists on disk.
    """
    info = meta["files"].get(input_file, {})
    sp_audio = info.get("split_audio")
    return bool(sp_audio and os.path.isfile(sp_audio))

def separate_already_done(meta: dict, input_file: str) -> bool:
    """
    Returns True if 'vocals' is in the metadata for this file
    AND the file physically exists on disk (same for background if you want).
    """
    info = meta["files"].get(input_file, {})
    v_path = info.get("vocals")
    return bool(v_path and os.path.isfile(v_path))

def diar_already_done(meta: dict, input_file: str) -> bool:
    """
    Returns True if 'diarization_data' is in the metadata for this file
    AND the diar_simple.json file physically exists on disk.
    """
    info = meta["files"].get(input_file, {})
    diar_path = info.get("diarization_data")
    return bool(diar_path and os.path.isfile(diar_path))

###############################################################################
# 5) process_single_file logic
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
    meta_path: str,
    num_speakers=None,
    language=None,
    device_index=0,
    metadata_dir=None
):
    """
    We fix the output_dir so each file gets a distinct subfolder:
      /folder_of_file/<base_name>/output_directory
    This ensures no overwriting.
    """
    file_dir = os.path.dirname(input_file)
    file_basename = os.path.splitext(os.path.basename(input_file))[0]
    output_dir = os.path.join(file_dir, file_basename, "output_directory")

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
        metadata_dir=metadata_dir
    )

    do_split = "split" in steps
    do_separate = "separate" in steps
    do_diar = "diar" in steps

    # Step: SPLIT
    if do_split:
        # Check if it was already done
        if split_already_done(meta, input_file):
            print(f"Skipping SPLIT for {input_file}, already done.")
        else:
            run_split(preprocessing, meta, meta_path)
    else:
        # If skipping, restore from meta if previously done
        prev_audio = meta["files"].get(input_file, {}).get("split_audio")
        if prev_audio and os.path.isfile(prev_audio):
            preprocessing.extracted_audio_path = prev_audio
        else:
            print(f"Warning: No existing split audio found for {input_file}. Step 'split' was skipped.")
            return

    # Step: SEPARATE
    if do_separate:
        # Check if it was already done
        if separate_already_done(meta, input_file):
            print(f"Skipping SEPARATE for {input_file}, already done.")
        else:
            run_separate(preprocessing, meta, meta_path)
    else:
        prev_vocals = meta["files"].get(input_file, {}).get("vocals")
        if prev_vocals and os.path.isfile(prev_vocals):
            preprocessing.vocals_path = prev_vocals
        else:
            print(f"Warning: No existing vocals file found for {input_file}. Step 'separate' was skipped.")
            return

    # Step: DIAR
    if do_diar:
        if diar_already_done(meta, input_file):
            print(f"Skipping DIAR for {input_file}, already done.")
        else:
            run_diar(preprocessing, meta, meta_path)

    print("\nFinal Output Paths:")
    paths = preprocessing.get_paths()
    for k, v in paths.items():
        print(f"{k}: {v}")

###############################################################################
# 6) Main
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Process audio/video files in separate steps.")
    parser.add_argument("input_path", type=str,
                        help="Path to a single file or directory containing files to process.")
    parser.add_argument("--extensions", type=str, nargs="*",
                        default=None,
                        help="One or more file extensions to process (e.g. mp3, mp4).")
    parser.add_argument("--audio_separator_model", type=str,
                        default=None,
                        help="Which model for audio separation.")
    parser.add_argument("--diarization_batch_size", type=int,
                        default=None,
                        help="Batch size for diarization.")
    parser.add_argument("--compute_type", type=str,
                        default=None,
                        help="WhisperX compute type: 'int8', 'float16', etc.")
    parser.add_argument("--HF_token", type=str,
                        default=None,
                        help="Hugging Face token if needed.")
    parser.add_argument("--config_file", type=str,
                        default=None,
                        help="Path to a config YAML file.")
    parser.add_argument("--metadata_file", type=str,
                        default="metadata.json",
                        help="Where to store/read the JSON metadata about splitted/separated/diarized files.")
    parser.add_argument("--steps", type=str,
                        default="split,separate,diar",
                        help="Comma-separated list of steps to run: [split, separate, diar].")
    parser.add_argument("--metadata_dir", type=str,
                        default=None,
                        help="Optional directory where segment metadata.json files should be saved. (Not used for segment-level metadata in this code)")

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
    if args.extensions:
        user_extensions = tuple("." + ext.strip(".").lower() for ext in args.extensions)
    else:
        user_extensions = None

    meta_path = args.metadata_file
    meta = load_metadata(meta_path)

    input_path = args.input_path
    files_to_process = []

    def is_valid_extension(fname: str, exts) -> bool:
        return fname.lower().endswith(exts)

    if os.path.isfile(input_path):
        files_to_process.append(os.path.abspath(input_path))
    else:
        for root, dirs, files in os.walk(input_path):
            if "output_directory" in root:
                continue
            for f in files:
                if user_extensions:
                    if is_valid_extension(f, user_extensions):
                        full_path = os.path.join(root, f)
                        files_to_process.append(os.path.abspath(full_path))
                else:
                    valid_exts = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac",
                                  ".mp4", ".mkv", ".mov")
                    if is_valid_extension(f, valid_exts):
                        full_path = os.path.join(root, f)
                        files_to_process.append(os.path.abspath(full_path))

    # If 'diar' is in steps, do the English->French priority
    if "diar" in steps_to_run:
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
            if english_path:
                print(f"\n--- Processing English file: {english_path} ---")
                process_single_file(
                    input_file=english_path,
                    steps=steps_to_run,
                    audio_separator_model=audio_separator_model,
                    diarization_batch_size=diarization_batch_size,
                    device=device,
                    compute_type=compute_type,
                    HF_token=HF_token,
                    meta=meta,
                    meta_path=meta_path,
                    language="en",
                    num_speakers=None,
                    device_index=device_index,
                    metadata_dir=args.metadata_dir
                )
                if "diar" in steps_to_run:
                    diar_json = meta["files"].get(english_path, {}).get("diarization_data")
                    if diar_json:
                        speaker_count = get_speaker_count(diar_json)
                        print(f"English speaker_count: {speaker_count}")

            for fpath in file_list:
                if fpath == english_path:
                    continue
                base = os.path.basename(fpath).lower()
                if base.startswith("vf"):
                    print(f"\n--- Processing French file: {fpath} ---")
                    process_single_file(
                        input_file=fpath,
                        steps=steps_to_run,
                        audio_separator_model=audio_separator_model,
                        diarization_batch_size=diarization_batch_size,
                        device=device,
                        compute_type=compute_type,
                        HF_token=HF_token,
                        meta=meta,
                        meta_path=meta_path,
                        language="fr",
                        num_speakers=speaker_count,
                        device_index=device_index,
                        metadata_dir=args.metadata_dir
                    )
                else:
                    print(f"\n--- Processing generic file: {fpath} ---")
                    process_single_file(
                        input_file=fpath,
                        steps=steps_to_run,
                        audio_separator_model=audio_separator_model,
                        diarization_batch_size=diarization_batch_size,
                        device=device,
                        compute_type=compute_type,
                        HF_token=HF_token,
                        meta=meta,
                        meta_path=meta_path,
                        device_index=device_index,
                        metadata_dir=args.metadata_dir
                    )
    else:
        for fpath in files_to_process:
            print(f"\n--- Processing file: {fpath} ---")
            process_single_file(
                input_file=fpath,
                steps=steps_to_run,
                audio_separator_model=audio_separator_model,
                diarization_batch_size=diarization_batch_size,
                device=device,
                compute_type=compute_type,
                HF_token=HF_token,
                meta=meta,
                meta_path=meta_path,
                device_index=device_index,
                metadata_dir=args.metadata_dir
            )

    print(f"\n=== All done! Updated metadata saved to: {meta_path} ===")


if __name__ == "__main__":
    main()