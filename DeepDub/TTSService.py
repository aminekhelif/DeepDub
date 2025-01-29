import os
import json
import yaml
import soundfile as sf
from omegaconf import OmegaConf
from loguru import logger

# --------------------- NEW: OpenVoice Imports ---------------------
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

# Load OpenVoice ToneColorConverter once
ckpt_converter = "/home/amine/OpenVoice/checkpoints_v2/converter"  # <-- Change to your path
device = "cuda:0" if torch.cuda.is_available() else "cpu"

logger.info("Loading ToneColorConverter (OpenVoice)...")
tone_color_converter = ToneColorConverter(f"{ckpt_converter}/config.json", device=device)
tone_color_converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")
logger.info("OpenVoice ToneColorConverter loaded.")


from f5_tts.infer.utils_infer import (
    load_model, infer_process, preprocess_ref_audio_text, load_vocoder,
)
from f5_tts.model import DiT

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

        vocoder_local = cfg.get("load_vocoder_from_local", False)
        self.vocoder = _load_vocoder_cli_style(self.vocoder_name, vocoder_local)

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
        remove_silence: bool = True
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
            target_rms= 0.01,
            cfg_strength=1,
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


logger.info("Initializing global TTS service...")
_f5_tts_service = _F5TTSService(_tts_cfg)
logger.info("TTS service is ready.")


def synthesize_translated_json(translated_json_path: str, temp_output_dir: str) -> str:
    """
    Public function: Reads the given 'translated' diarization JSON (which
    must have fields 'audio_path', 'text', 'translated_text', 'duration'),
    runs TTS for each segment, and outputs a new JSON with 'tts' field.

    The final .wav files go to something like /home/.../tts_output/{i}.wav
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

    # ------------------ CHANGE #1: Use temp_output_dir as-is ------------------
    tts_dir = temp_output_dir
    os.makedirs(tts_dir, exist_ok=True)

    for i, seg in enumerate(segments):
        ref_audio = seg.get("audio_path")
        ref_text = seg.get("text")
        new_text = seg.get("translated_text")
        seg_dur  = seg.get("duration", 1)
        seg_dur += 0.5

        if not ref_audio or not ref_text or not new_text:
            logger.warning(f"[TTS] Segment {i} missing fields, skipping.")
            continue

        # Keep your original line for adding dots:
        new_text = seg.get("translated_text") + " ..."

        out_wav = os.path.join(tts_dir, f"{i}.wav")
        try:
            # ----------- (1) Run F5-TTS -----------
            length_s = _f5_tts_service.run_inference(
                ref_audio_path=ref_audio,
                ref_text=ref_text,
                gen_text=new_text,
                output_wav=out_wav,
                fix_duration=seg_dur,
                remove_silence=False,
            )

            # ----------- (2) Run OpenVoice Timbre Transfer -----------
            speaker_id = seg.get("speaker")
            if speaker_id:
                # ------------------ CHANGE #2: Go up one directory -------------
                # e.g. if ref_audio is:
                #   /home/amine/.../SPEAKER_08/segment_1/audio.wav
                # then the concatenated file is in:
                #   /home/amine/.../SPEAKER_08/SPEAKER_08_concatenated.wav
                speaker_dir = os.path.dirname(os.path.dirname(ref_audio))
                concatenated_ref = os.path.join(speaker_dir, f"{speaker_id}_concatenated.wav")

                if os.path.exists(concatenated_ref):
                    target_se, _ = se_extractor.get_se(concatenated_ref, tone_color_converter, vad=True)
                    source_se, _ = se_extractor.get_se(out_wav, tone_color_converter, vad=True)

                    tone_color_converter.convert(
                        audio_src_path=out_wav,
                        src_se=source_se,
                        tgt_se=target_se,
                        output_path=out_wav
                    )
                    logger.info(f"[OpenVoice] Timbre transfer done => {out_wav}")
                else:
                    logger.warning(f"[OpenVoice] Missing concatenated file {concatenated_ref}, skipping timbre transfer.")
            else:
                logger.warning(f"[OpenVoice] Segment {i} has no speaker, skipping timbre transfer.")

            # Store results in JSON
            seg["tts"] = {
                "audio_path": out_wav,
                "duration": length_s
            }

        except Exception as e:
            logger.error(f"[TTS] Segment {i} error: {e}")
            seg["tts"] = {"error": str(e)}

    out_json = os.path.splitext(translated_json_path)[0] + "_tts.json"
    with open(out_json, "w", encoding="utf-8") as wf:
        json.dump(segments, wf, indent=4, ensure_ascii=False)

    logger.info(f"[TTS] Updated => {out_json}")
    return json.dumps(segments, indent=4, ensure_ascii=False)