# DeepDub/TTSService.py

import os
import json
import yaml
import soundfile as sf
from omegaconf import OmegaConf
from loguru import logger

# F5-TTS imports
from f5_tts.infer.utils_infer import (
    load_model, infer_process, preprocess_ref_audio_text, load_vocoder,
)
from f5_tts.model import DiT

# 1) Load the main config.yaml
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as cf:
    _config = yaml.safe_load(cf)

_tts_cfg = _config["tts"] 
def _load_vocoder_cli_style(vocoder_name: str, load_local: bool):
    """Helper to load the vocoder from local or huggingface."""
    logger.info(f"Loading vocoder: {vocoder_name}, local={load_local}")
    return load_vocoder(vocoder_name, is_local=load_local)

class _F5TTSService:
    """
    Internal class that loads F5-TTS + vocoder once, so you can
    call run_inference(...) multiple times.
    """
    def __init__(self, cfg: dict):
        self.device = cfg.get("device", "cuda")
        self.nfe_step = cfg.get("nfe_step", 32)
        self.vocoder_name = cfg.get("vocoder_name", "vocos")

        # 1) Load vocoder
        vocoder_local = cfg.get("load_vocoder_from_local", False)
        self.vocoder = _load_vocoder_cli_style(self.vocoder_name, vocoder_local)

        # 2) Load F5-TTS model
        ckpt_file = cfg["ckpt_file"]
        vocab_file = cfg["vocab_file"]
        model_cfg_path = cfg["model_cfg_path"]

        logger.info(f"Loading F5-TTS from {ckpt_file}")
        model_arch = OmegaConf.load(model_cfg_path).model.arch
        self.ema_model = load_model(
            DiT,
            model_arch,
            ckpt_file,
            mel_spec_type=self.vocoder_name,
            vocab_file=vocab_file,
            device=self.device,
        )
        logger.info("F5-TTS model loaded successfully.")

    def run_inference(
        self,
        ref_audio_path: str,
        ref_text: str,
        gen_text: str,
        output_wav: str,
        fix_duration: float,
        remove_silence: bool = False
    ) -> float:
        """
        Generates speech matching fix_duration. Returns the wave length (sec).
        """
        logger.info(f"[TTS] ref_audio='{ref_audio_path}', text='{ref_text}', gen='{gen_text}'")

        ref_audio_proc, ref_text_proc = preprocess_ref_audio_text(ref_audio_path, ref_text)
        audio_segment, final_sr, _ = infer_process(
            ref_audio=ref_audio_proc,
            ref_text=ref_text_proc,
            gen_text=gen_text,
            model_obj=self.ema_model,
            vocoder=self.vocoder,
            mel_spec_type=self.vocoder_name,
            nfe_step=self.nfe_step,
            fix_duration=fix_duration,
            device=self.device,
        )

        os.makedirs(os.path.dirname(output_wav), exist_ok=True)
        sf.write(output_wav, audio_segment, final_sr, format="WAV")
        duration = len(audio_segment) / final_sr
        logger.info(f"[TTS] Output => {output_wav} (duration={duration:.2f}s)")

        if remove_silence:
            from f5_tts.infer.utils_infer import remove_silence_for_generated_wav
            remove_silence_for_generated_wav(output_wav)

        return duration

# Create one global TTS service at import
logger.info("Initializing global TTS service...")
_f5_tts_service = _F5TTSService(_tts_cfg)
logger.info("TTS service is ready.")


def synthesize_translated_json(translated_json_path: str) -> str:
    """
    Public function: Reads the given 'translated' diarization JSON (which
    must have fields 'audio_path', 'text', 'translated_text', 'duration'),
    runs TTS for each segment, and outputs a new JSON with 'tts' field.

    The final .wav files go to /tmp/deepdub_outputs/tts_output/{i}.wav
    Returns the updated JSON as a string for easy display.
    """
    if not os.path.exists(translated_json_path):
        msg = f"Error: file not found -> {translated_json_path}"
        logger.error(msg)
        return msg

    with open(translated_json_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    if not isinstance(segments, list):
        msg = "Error: JSON is not a list of segments."
        logger.error(msg)
        return msg

    tts_dir = "/tmp/deepdub_outputs/tts_output"
    os.makedirs(tts_dir, exist_ok=True)

    for i, seg in enumerate(segments):
        ref_audio = seg.get("audio_path")
        ref_text = seg.get("text")
        new_text = seg.get("translated_text")
        seg_dur  = seg.get("duration", 0.0)
        if not ref_audio or not ref_text or not new_text:
            logger.warning(f"[TTS] Segment {i} missing fields, skipping.")
            continue

        out_wav = os.path.join(tts_dir, f"{i}.wav")
        try:
            length_s = _f5_tts_service.run_inference(
                ref_audio_path=ref_audio,
                ref_text=ref_text,
                gen_text=new_text,
                output_wav=out_wav,
                fix_duration=seg_dur,
                remove_silence=False,
            )
            seg["tts"] = {
                "audio_path": out_wav,
                "duration": length_s
            }
        except Exception as e:
            logger.error(f"[TTS] Segment {i} error: {e}")
            seg["tts"] = {"error": str(e)}

    # Save an updated file
    out_json = os.path.splitext(translated_json_path)[0] + "_tts.json"
    with open(out_json, "w", encoding="utf-8") as wf:
        json.dump(segments, wf, indent=4, ensure_ascii=False)

    logger.info(f"[TTS] Updated => {out_json}")
    return json.dumps(segments, indent=4, ensure_ascii=False)