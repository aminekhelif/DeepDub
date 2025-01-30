import os
import glob
import json
from copy import deepcopy
from typing import List, Dict, Any


def load_segments_from_json(
    json_path: str,
    relative_subfolder: str,
    skip_empty_or_placeholder: bool = True
) -> List[Dict[str, Any]]:
    """
    Loads segments from a JSON file containing a list of dicts, each with
    at least 'start' and 'end' keys, optionally 'text' and 'speaker'.

    For each segment, we store:
      - "start", "end", "duration"
      - "text": a string
      - "speaker": a list of length 1 (or ["Unknown"] if missing)
      - "indexes": a list with the original index
      - "sub_segments": a list of dicts => [ {"start":..., "end":...} ]
      - "audio_paths": a list with exactly one relative path to the audio,
        e.g. "/VF/output_directory/diarization/speaker_audio/SPEAKER_00/segment_3/audio.wav"

    If skip_empty_or_placeholder = True, we skip segments whose text is empty
    or matches certain placeholders (case-insensitive).
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    segments = []
    placeholders = {
        "...", "... ...", "... ... ...", "... ... ... ...",
        "thank you", "sous-titrage", "titrage"
    }

    for i, seg in enumerate(data):
        text_value = seg.get("text", "").strip()
        lower_text = text_value.lower()

        if skip_empty_or_placeholder:
            if not lower_text or any(ph in lower_text for ph in placeholders):
                continue

        start = seg["start"]
        end = seg["end"]
        duration = end - start

        raw_speaker = seg.get("speaker", "")
        if not raw_speaker:
            raw_speaker = "Unknown"

        speaker_list = [raw_speaker]
        index_list   = [i]
        sub_segments = [{"start": start, "end": end}]

        audio_folder_name = raw_speaker
        audio_path = (
            f"/{relative_subfolder}/output_directory/diarization/speaker_audio/"
            f"{audio_folder_name}/segment_{i}/audio.wav"
        )
        audio_paths_list = [audio_path]

        seg_dict = {
            "start": start,
            "end": end,
            "duration": duration,
            "text": text_value,
            "speaker": speaker_list,
            "indexes": index_list,
            "sub_segments": sub_segments,
            "audio_paths": audio_paths_list
        }
        segments.append(seg_dict)

    return segments


def merge_small_segments(
    segments: List[Dict[str, Any]],
    min_merge_duration: float = 1.0,
    max_gap_before_merge: float = 0.3,
    merge_speakers: bool = False
) -> List[Dict[str, Any]]:
    """
    Merge consecutive segments if:
      - At least one is 'short' (< min_merge_duration)
      - They are close (gap <= max_gap_before_merge)
      - If merge_speakers=False, they have identical speaker-lists
        If merge_speakers=True, ignore speaker differences.

    When merging, we:
      - Combine text (with a space)
      - Append speaker/indexes/sub_segments/audio_paths
      - Update 'start','end','duration'
    """
    if not segments:
        return []

    segments = sorted(segments, key=lambda x: x["start"])
    merged = []
    buffer_seg = None

    for seg in segments:
        if buffer_seg is None:
            buffer_seg = deepcopy(seg)
        else:
            gap = seg["start"] - buffer_seg["end"]
            short_buffer  = (buffer_seg["duration"] < min_merge_duration)
            short_current = (seg["duration"] < min_merge_duration)
            close_enough  = (0 <= gap <= max_gap_before_merge)

            if not merge_speakers:
                same_spk = (buffer_seg["speaker"] == seg["speaker"])
            else:
                same_spk = True

            if (short_buffer or short_current) and close_enough and same_spk:
                new_start = min(buffer_seg["start"], seg["start"])
                new_end   = max(buffer_seg["end"], seg["end"])
                new_duration = new_end - new_start

                merged_text = (buffer_seg["text"] + " " + seg["text"]).strip()

                merged_speakers  = buffer_seg["speaker"] + seg["speaker"]
                merged_indexes   = buffer_seg["indexes"] + seg["indexes"]
                merged_subs      = buffer_seg["sub_segments"] + seg["sub_segments"]
                merged_audio     = buffer_seg["audio_paths"] + seg["audio_paths"]

                buffer_seg["start"]       = new_start
                buffer_seg["end"]         = new_end
                buffer_seg["duration"]    = new_duration
                buffer_seg["text"]        = merged_text
                buffer_seg["speaker"]     = merged_speakers
                buffer_seg["indexes"]     = merged_indexes
                buffer_seg["sub_segments"] = merged_subs
                buffer_seg["audio_paths"]  = merged_audio
            else:
                merged.append(buffer_seg)
                buffer_seg = deepcopy(seg)

    if buffer_seg is not None:
        merged.append(buffer_seg)

    return merged


def compute_time_intersection(e_start, e_end, f_start, f_end) -> float:
    overlap_start = max(e_start, f_start)
    overlap_end   = min(e_end, f_end)
    return max(0.0, overlap_end - overlap_start)

def measure_overlap_ratio(
    e_start, e_end,
    f_start, f_end,
    mode: str = "english"
) -> float:
    """
    mode='english': intersection / english_duration
    mode='french':  intersection / french_duration
    mode='iou':     intersection / union of durations
    """
    intersection = compute_time_intersection(e_start, e_end, f_start, f_end)
    e_len = e_end - e_start
    f_len = f_end - f_start

    if e_len <= 0 or f_len <= 0:
        return 0.0

    if mode == "english":
        return intersection / e_len
    elif mode == "french":
        return intersection / f_len
    elif mode == "iou":
        union = e_len + f_len - intersection
        return intersection / union if union > 0 else 0.0
    else:
        raise ValueError(f"Unknown overlap mode: {mode}")


def align_segments(
    english_segments: List[Dict[str, Any]],
    french_segments: List[Dict[str, Any]],
    overlap_threshold: float = 0.8,
    overlap_mode: str = "english"
) -> List[Dict[str, Any]]:
    """
    Two-pointer alignment: if overlap ratio >= threshold => match, then advance.

    Returns a list of matches, each a dict:
      {
        "english": { ... }, "french": { ... }, "overlap_ratio": <float>
      }
    """
    i, j = 0, 0
    matches = []

    while i < len(english_segments) and j < len(french_segments):
        e_seg = english_segments[i]
        f_seg = french_segments[j]

        e_start, e_end = e_seg["start"], e_seg["end"]
        f_start, f_end = f_seg["start"], f_seg["end"]

        ratio = measure_overlap_ratio(e_start, e_end, f_start, f_end, mode=overlap_mode)

        if ratio >= overlap_threshold:
            e_duration = round(e_end - e_start, 3)
            f_duration = round(f_end - f_start, 3)

            matches.append({
                "english": {
                    "segment_indexes": e_seg["indexes"],
                    "speakers": e_seg["speaker"],
                    "start": e_start,
                    "end": e_end,
                    "duration": e_duration,
                    "text": e_seg["text"],
                    "sub_segments": e_seg["sub_segments"],
                    "audio_paths": e_seg["audio_paths"]
                },
                "french": {
                    "segment_indexes": f_seg["indexes"],
                    "speakers": f_seg["speaker"],
                    "start": f_start,
                    "end": f_end,
                    "duration": f_duration,
                    "text": f_seg["text"],
                    "sub_segments": f_seg["sub_segments"],
                    "audio_paths": f_seg["audio_paths"]
                },
                "overlap_ratio": round(ratio, 3)
            })

            if e_end < f_end:
                i += 1
            elif f_end < e_end:
                j += 1
            else:
                i += 1
                j += 1
        else:
            if e_end < f_end:
                i += 1
            else:
                j += 1

    return matches


def enhanced_demo_alignment(
    english_json_path: str,
    french_json_path: str,
    do_merge: bool = True,
    min_merge_duration: float = 1.0,
    max_gap_before_merge: float = 0.3,
    merge_speakers: bool = False,
    overlap_threshold: float = 0.8,
    overlap_mode: str = "english",
    output_path: str = "aligned_output.json"
):
    """
    1) Detect subfolder (VF/VFF/VO_anglais) from path,
       load segments with relative audio_paths.
    2) Optionally merge short segments.
    3) Align them.
    4) Print summary, save matched pairs to JSON.
    """
    def detect_subfolder(path: str) -> str:
        lowered = path.lower()
        if "/vf/" in lowered:
            return "VF"
        elif "/vff/" in lowered:
            return "VFF"
        elif "/vo_anglais/" in lowered:
            return "VO_anglais"
        return "UnknownLang"

    eng_subfolder = detect_subfolder(english_json_path)
    fr_subfolder  = detect_subfolder(french_json_path)

    eng_segments = load_segments_from_json(
        english_json_path, relative_subfolder=eng_subfolder
    )
    fr_segments  = load_segments_from_json(
        french_json_path, relative_subfolder=fr_subfolder
    )

    print(f"Loaded {len(eng_segments)} English segments, {len(fr_segments)} French segments.")

    if do_merge:
        eng_segments = merge_small_segments(
            eng_segments,
            min_merge_duration=min_merge_duration,
            max_gap_before_merge=max_gap_before_merge,
            merge_speakers=merge_speakers
        )
        fr_segments = merge_small_segments(
            fr_segments,
            min_merge_duration=min_merge_duration,
            max_gap_before_merge=max_gap_before_merge,
            merge_speakers=merge_speakers
        )
        print(f"After merging => {len(eng_segments)} English segments, "
              f"{len(fr_segments)} French segments.")

    matches = align_segments(
        eng_segments, fr_segments,
        overlap_threshold=overlap_threshold,
        overlap_mode=overlap_mode
    )

    n_eng = len(eng_segments)
    n_fr  = len(fr_segments)
    matched_pairs = len(matches)
    total_overlap = sum(m["overlap_ratio"] for m in matches)
    avg_overlap = (total_overlap / matched_pairs) if matched_pairs else 0.0

    print("\n===== Alignment Summary =====")
    print(f"English segments: {n_eng}")
    print(f"French segments:  {n_fr}")
    print(f"Matched pairs:    {matched_pairs}")
    print(f"Average overlap ratio (mode={overlap_mode}): {avg_overlap:.3f}")

    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(matches, f_out, indent=2, ensure_ascii=False)

    print(f"\nWrote alignment results to {output_path}.\n")



if __name__ == "__main__":
    import sys

    min_merge_durations = 0.5
    max_gap_before_merges = 4.0
    merge_speakers_options = True
    overlap_modes = "iou"
    overlap_threshold = 0.8

    base_dir = "Data"

    merge_matched_segments = False
    if len(sys.argv) > 1 and sys.argv[1].lower() == "merge_all":
        merge_matched_segments = True

    film_dirs = sorted(glob.glob(os.path.join(base_dir, "FILM*")))

    for film_dir in film_dirs:
        vf_json = os.path.join(film_dir, "VF", "output_directory", "diarization", "diar_simple.json")
        if not os.path.exists(vf_json):
            vf_json_alt = os.path.join(film_dir, "VFF", "output_directory", "diarization", "diar_simple.json")
            if os.path.exists(vf_json_alt):
                vf_json = vf_json_alt

        vo_json = os.path.join(film_dir, "VO_anglais", "output_directory", "diarization", "diar_simple.json")

        if not os.path.exists(vf_json):
            print(f"[WARNING] No VF or VFF diar_simple.json found for {film_dir}. Skipping.")
            continue
        if not os.path.exists(vo_json):
            print(f"[WARNING] No VO_anglais diar_simple.json found for {film_dir}. Skipping.")
            continue

        print(f"\n=== Processing film: {film_dir} ===")
        print(f"   French JSON:  {vf_json}")
        print(f"   English JSON: {vo_json}")

        output_path = os.path.join(film_dir, "matched_segments.json")

        enhanced_demo_alignment(
            english_json_path=vo_json,
            french_json_path=vf_json,
            do_merge=True,
            min_merge_duration=min_merge_durations,
            max_gap_before_merge=max_gap_before_merges,
            merge_speakers=merge_speakers_options,
            overlap_threshold=overlap_threshold,
            overlap_mode=overlap_modes,
            output_path=output_path
        )


    if merge_matched_segments:
        all_matches = []

        for film_dir in film_dirs:
            matched_file = os.path.join(film_dir, "matched_segments.json")
            if not os.path.exists(matched_file):
                continue

            film_name = os.path.basename(film_dir)

            print(f"[Merging] {matched_file} into global JSON. Prefixing audio paths with /{film_name} ...")

            with open(matched_file, 'r', encoding='utf-8') as f_in:
                matches = json.load(f_in)

            for match in matches:
                audio_paths = match["english"]["audio_paths"]
                new_audio_paths = []
                for path in audio_paths:
                    new_audio_paths.append(f"/{film_name}{path}") 
                match["english"]["audio_paths"] = new_audio_paths

                audio_paths = match["french"]["audio_paths"]
                new_audio_paths = []
                for path in audio_paths:
                    new_audio_paths.append(f"/{film_name}{path}") 
                match["french"]["audio_paths"] = new_audio_paths

                all_matches.append(match)

        merged_output = os.path.join(base_dir, "merged_all_films.json")
        with open(merged_output, 'w', encoding='utf-8') as f_out:
            json.dump(all_matches, f_out, indent=2, ensure_ascii=False)

        print(f"\n[Done] Wrote merged JSON of all films to: {merged_output}\n")