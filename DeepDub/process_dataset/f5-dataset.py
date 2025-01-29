#!/usr/bin/env python3
import os
import json
import torch
import torchaudio
import soundfile as sf

from pathlib import Path
from loguru import logger
from tqdm import tqdm

# -------------------------------------------------------------------
#   Imports from your F5-TTS codebase (unchanged)
# -------------------------------------------------------------------
from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,                # We'll pass the checkpoint path as the 3rd positional arg
    preprocess_ref_audio_text,
    mel_spec_type as default_mel_spec_type,
    target_rms as default_target_rms,
    cross_fade_duration as default_cross_fade_duration,
    cfg_strength as default_cfg_strength,
    sway_sampling_coef as default_sway_sampling_coef,
    speed as default_speed,
    fix_duration as default_fix_duration,
    nfe_step as default_nfe_step,
)
from f5_tts.model import DiT
from omegaconf import OmegaConf


def load_vocoder_cli_style(
    vocoder_name: str,
    load_vocoder_from_local: bool,
    local_path_vocos: str = "../checkpoints/vocos-mel-24khz",
    local_path_bigvgan: str = "../checkpoints/bigvgan_v2_24khz_100band_256x",
):
    from f5_tts.infer.utils_infer import load_vocoder

    if vocoder_name == "vocos":
        if load_vocoder_from_local and os.path.isdir(local_path_vocos):
            print(f"Load vocos from local path {local_path_vocos}")
            return load_vocoder(
                vocoder_name="vocos",
                is_local=True,
                local_path=local_path_vocos,
            )
        else:
            print("Download Vocos from huggingface charactr/vocos-mel-24khz")
            return load_vocoder(
                vocoder_name="vocos",
                is_local=False,
                local_path=local_path_vocos,
            )
    elif vocoder_name == "bigvgan":
        if load_vocoder_from_local and os.path.isdir(local_path_bigvgan):
            print(f"Load BigVGAN from local path {local_path_bigvgan}")
            return load_vocoder(
                vocoder_name="bigvgan",
                is_local=True,
                local_path=local_path_bigvgan,
            )
        else:
            print("Download BigVGAN from huggingface ???")
            return load_vocoder(
                vocoder_name="bigvgan",
                is_local=False,
                local_path=local_path_bigvgan,
            )
    else:
        raise ValueError(f"Unknown vocoder_name={vocoder_name}")


# -------------------------------------------------------------------
#   F5TTSService: loads F5-TTS model + vocoder once, does inference
# -------------------------------------------------------------------
class F5TTSService:
    """
    Loads the F5-TTS model + vocoder once, replicating your CLI logic,
    so you can do run_inference(...) multiple times.
    """
    def __init__(
        self,
        ckpt_file: str,
        vocab_file: str,
        model_cfg_path: str,
        vocoder_name: str = "vocos",
        load_vocoder_from_local: bool = True,
        device: str = "cuda",
        target_rms: float = default_target_rms,
        cross_fade_duration: float = default_cross_fade_duration,
        nfe_step: int = 32,
        cfg_strength: float = default_cfg_strength,
        sway_sampling_coef: float = default_sway_sampling_coef,
        speed: float = default_speed,
    ):
        self.device = device
        self.ckpt_file = ckpt_file
        self.vocab_file = vocab_file
        self.model_cfg_path = model_cfg_path
        self.vocoder_name = vocoder_name
        self.load_vocoder_from_local = load_vocoder_from_local

        self.target_rms = target_rms
        self.cross_fade_duration = cross_fade_duration
        self.nfe_step = nfe_step
        self.cfg_strength = cfg_strength
        self.sway_sampling_coef = sway_sampling_coef
        self.speed = speed

        # 1) Load vocoder from CLI fallback
        self.vocoder = load_vocoder_cli_style(
            vocoder_name=self.vocoder_name,
            load_vocoder_from_local=self.load_vocoder_from_local,
        )

        # 2) Load F5-TTS model
        print("Using F5-TTS...")
        print(f"vocab :  {self.vocab_file}")
        print(f"token :  custom")
        print(f"model :  {self.ckpt_file}\n")

        self.model_cfg = OmegaConf.load(self.model_cfg_path).model.arch

        # pass ckpt_file as the 3rd positional argument => ckpt_path
        self.ema_model = load_model(
            DiT,
            self.model_cfg,
            self.ckpt_file,        # correct positional param for ckpt_path
            mel_spec_type=self.vocoder_name,
            vocab_file=self.vocab_file,
            device=self.device,
        )

        print("F5-TTS model loaded. Ready for inference.\n")

    def run_inference(
        self,
        merged_ref_audio_path: str,
        ref_text: str,
        gen_text: str,
        output_wav: str,
        fix_duration: float = 0.0,
        remove_silence: bool = False,
    ) -> float:
        """
        1) Preprocess reference audio + text
        2) call infer_process(...)
        3) Save final wave => output_wav (with format="WAV" so .wav.tmp is recognized)
        4) Return the audio duration
        """
        print("Voice: main")
        print("ref_audio ", merged_ref_audio_path)
        print("Converting audio...")
        print("Using custom reference text...\n")

        ref_audio_proc, ref_text_proc = preprocess_ref_audio_text(merged_ref_audio_path, ref_text)
        print("ref_text  ", ref_text, "\nref_audio_", ref_audio_proc, "\n\n")

        print("No voice tag found, using main.")
        print("Voice: main")
        print("gen_text 0", gen_text)
        print("\nGenerating audio in 1 batches...")

        audio_segment, final_sr, _ = infer_process(
            ref_audio=ref_audio_proc,
            ref_text=ref_text_proc,
            gen_text=gen_text,
            model_obj=self.ema_model,
            vocoder=self.vocoder,
            mel_spec_type=self.vocoder_name,
            target_rms=self.target_rms,
            cross_fade_duration=self.cross_fade_duration,
            nfe_step=self.nfe_step,
            cfg_strength=self.cfg_strength,
            sway_sampling_coef=self.sway_sampling_coef,
            speed=self.speed,
            fix_duration=fix_duration,
            device=self.device,
        )

        os.makedirs(os.path.dirname(output_wav), exist_ok=True)

        # >>> ADD format="WAV" so PySoundFile recognizes .wav.tmp extension
        sf.write(output_wav, audio_segment, final_sr, format="WAV")
        duration = len(audio_segment) / final_sr
        print(f"{output_wav}")

        if remove_silence:
            from f5_tts.infer.utils_infer import remove_silence_for_generated_wav
            remove_silence_for_generated_wav(output_wav)

        return duration


# -------------------------------------------------------------------
#  create_silence: wave of zeros for "duration_sec"
# -------------------------------------------------------------------
def create_silence(duration_sec: float, sr: int) -> torch.Tensor:
    length = int(round(duration_sec * sr))
    return torch.zeros((1, length), dtype=torch.float32)

# -------------------------------------------------------------------
#  merge_subsegments_dyn: merges ANY number of sub-segments
# -------------------------------------------------------------------
def merge_subsegments_dyn(
    sub_paths: list[str],
    total_duration: float,
    base_dir: str
) -> (str, float):
    """
    Load each sub-wave => sum_sub => total wave length
    gap = total_duration - sum_sub
    Distribute gap across (N+1) => final= [sil, sub1, sil, sub2, ... subN, sil]
    """
    import torch
    if not sub_paths:
        logger.warning("No sub-segments found!")
        return "empty_ref.wav", 16000

    waves = []
    sample_rate = None
    sum_sub = 0.0
    for path in sub_paths:
        full_path = os.path.join(base_dir, path.lstrip("/"))
        wav, sr = torchaudio.load(full_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sample_rate is None:
            sample_rate = sr
        else:
            if sr != sample_rate:
                logger.warning(f"Resampling from {sr} to {sample_rate}")
                wav = torchaudio.functional.resample(wav, sr, sample_rate)

        length_sec = wav.shape[-1] / sample_rate
        sum_sub += length_sec
        waves.append(wav)

    gap = total_duration - sum_sub
    if gap < 0:
        logger.warning(
            f"gap < 0 for total_duration={total_duration:.2f}, sum_sub={sum_sub:.2f}, set gap=0"
        )
        gap = 0.0

    N = len(waves)
    chunk = gap / (N + 1) if N > 0 else gap
    sil_chunk = create_silence(chunk, sample_rate)

    # Build final wave => [sil, sub1, sil, sub2, ..., subN, sil]
    final_list = [sil_chunk]
    for wav in waves:
        final_list.append(wav)
        final_list.append(sil_chunk)

    merged = torch.cat(final_list, dim=-1)

    merged_path = "merged_ref.wav"
    sf.write(merged_path, merged.squeeze(0).numpy(), sample_rate)
    return merged_path, sample_rate


# -------------------------------------------------------------------
#  MAIN SCRIPT
# -------------------------------------------------------------------
if __name__ == "__main__":

    # 1) Input/Output
    json_path = "/home/amine/DeepDub/Data/merged_all_films_with_f5.json"
    output_json_path = "/home/amine/DeepDub/Data/merged_all_films_with_f5-64.json"
    f5_folder = "/home/amine/DeepDub/Data/f5-64"
    os.makedirs(f5_folder, exist_ok=True)

    # 2) Initialize once
    service = F5TTSService(
        ckpt_file="/home/amine/DeepDub/F5-TTS/ckpts/model_last.pt",
        vocab_file="/home/amine/DeepDub/F5-TTS/ckpts/vocab.txt",
        model_cfg_path="/home/amine/DeepDub/F5-TTS/src/f5_tts/configs/F5TTS_Base_train.yaml",
        vocoder_name="vocos",
        load_vocoder_from_local=False,
        device="cuda",
        nfe_step=32,
    )

    # 3) Load JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    base_dir = os.path.dirname(json_path)

    total_rows = len(data)
    logger.info(f"Found {total_rows} total segments in JSON.")

    # 4) Determine missing rows => i => i.wav
    missing_rows = []
    for i in range(total_rows):
        final_wav = os.path.join(f5_folder, f"{i}.wav")
        if not os.path.exists(final_wav):
            missing_rows.append(i)

    logger.info(f"{len(missing_rows)} missing .wav from {total_rows} total.")

    # 5) Process missing
    for i in tqdm(missing_rows, desc="Processing missing rows"):
        seg = data[i]
        english_info = seg["english"]
        fr_text = seg["french"]["text"]

        final_wav = os.path.join(f5_folder, f"{i}.wav")
        tmp_wav   = final_wav + ".tmp"  # partial safety

        eng_duration = english_info["duration"]
        eng_text     = english_info["text"]
        audio_paths  = english_info["audio_paths"]

        # Merge sub-segments if multiple
        merged_ref, sr_ = merge_subsegments_dyn(audio_paths, eng_duration, base_dir)

        logger.info(f"[F5] Row={i}, fix_duration={eng_duration:.2f}s => {tmp_wav}")
        dur = service.run_inference(
            merged_ref_audio_path=merged_ref,
            ref_text=eng_text,
            gen_text=fr_text,
            output_wav=tmp_wav,  # we pass .tmp
            fix_duration=eng_duration,
            remove_silence=False
        )

        # If .tmp was successfully created, rename => final
        if os.path.exists(tmp_wav):
            os.rename(tmp_wav, final_wav)
        else:
            logger.warning(f"No .tmp file for row {i}, skipping rename.")
            continue

        # Update JSON
        seg.setdefault("tts", {})
        seg["tts"]["f5-tts-64"] = {
            "audio_path": f"/f5-64/{i}.wav",
            "duration": dur
        }

    # 6) Save updated JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Done! Updated JSON => {output_json_path}")