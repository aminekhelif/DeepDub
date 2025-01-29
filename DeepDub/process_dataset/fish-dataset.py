#!/usr/bin/env python3
import os
import json
import soundfile as sf
from pathlib import Path
from loguru import logger
from tqdm import tqdm

# ---------------------------------------------------------------
#  >>> HYDRA / VQGAN code (replicates your vqgan/inference.py) <<<
# ---------------------------------------------------------------
import torch
import torchaudio

import hydra
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

def load_vqgan_model(config_name: str, checkpoint_path: str, device: str = "cuda"):
    import hydra.core.global_hydra

    hydra.core.global_hydra.GlobalHydra.instance().clear()

    config_dir = "/home/amine/DeepDub/fisher/fish_speech/fish_speech/configs"
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name=config_name)

    model = instantiate(cfg)
    state_dict = torch.load(checkpoint_path, map_location=device, mmap=True, weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if any("generator." in k for k in state_dict):
        filtered = {}
        for k, v in state_dict.items():
            if "generator." in k:
                new_k = k.replace("generator.", "")
                filtered[new_k] = v
        state_dict = filtered

    result = model.load_state_dict(state_dict, strict=False, assign=True)
    model.eval().to(device)
    logger.info(f"[VQGAN] Loaded model with result: {result}")
    return model

@torch.no_grad()
def vqgan_encode_audio(vqgan_model, input_wav: str, output_tensor: str, device: str = "cuda"):
    audio, sr = torchaudio.load(input_wav)
    if audio.shape[0] > 1:
        # stereo -> mono
        audio = audio.mean(dim=0, keepdim=True)
    audio = torchaudio.functional.resample(audio, sr, vqgan_model.spec_transform.sample_rate).to(device)

    audio_lengths = torch.tensor([audio.shape[1]], dtype=torch.long, device=device)
    batch_audio = audio.unsqueeze(0)
    indices = vqgan_model.encode(batch_audio, audio_lengths)[0][0]
    torch.save(indices.cpu(), output_tensor)
    logger.info(f"[VQGAN] Encoded {input_wav} -> {output_tensor}")

@torch.no_grad()
def vqgan_decode_codes(vqgan_model, input_tensor: str, output_wav: str, device: str = "cuda"):
    codes_t = torch.load(input_tensor).to(device)
    if codes_t.ndim != 2:
        raise ValueError(f"[VQGAN] Expected shape [n_codebooks, T], got {codes_t.shape}")

    batch_codes = codes_t.unsqueeze(0)
    lengths = torch.tensor([codes_t.shape[1]], device=device)
    fake_audio, _ = vqgan_model.decode(indices=batch_codes, feature_lengths=lengths)
    fake_audio_np = fake_audio[0, 0].cpu().numpy()

    sr = vqgan_model.spec_transform.sample_rate
    sf.write(output_wav, fake_audio_np, sr)
    logger.info(f"[VQGAN] Decoded {input_tensor} -> {output_wav} (SR={sr})")

# ---------------------------------------------------------------
#  >>> TEXT2SEMANTIC code (replicates your text2semantic/inference.py) <<<
#     WITH a fix for dimension mismatch by truncating prompts
# ---------------------------------------------------------------
from fish_speech.models.text2semantic.inference import (
    load_model as load_text2semantic_model,
    generate_long as original_generate_long,
    GenerateResponse,
)

import torch

# >>> We import the original 'generate' to call after truncation
from fish_speech.models.text2semantic.inference import generate as real_generate

def patched_generate(
    *,
    model,
    prompt: torch.Tensor,
    max_new_tokens: int,
    decode_one_token,
    temperature=0.7,
    top_p=0.7,
    repetition_penalty=1.2,
    **sampling_kwargs,
):
    """
    A patched version of generate() that ensures prompt doesn't exceed max_seq_len,
    then calls the original generate() with the truncated prompt.
    """
    from loguru import logger
    from fish_speech.tokenizer import IM_END_TOKEN

    T = prompt.size(1)
    codebook_dim = 1 + model.config.num_codebooks
    max_len = model.config.max_seq_len

    if T > max_len:
        logger.warning(f"[PATCHED] Prompt length {T} exceeds max_seq_len={max_len}. Truncating.")
        prompt = prompt[:, :max_len]
        T = max_len

    # >>> Instead of raising NotImplementedError, we call the real generate
    return real_generate(
        model=model,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        decode_one_token=decode_one_token,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        **sampling_kwargs,
    )


def patched_generate_long(
    *,
    model,
    device: str | torch.device,
    decode_one_token: callable,
    text: str,
    num_samples: int = 1,
    max_new_tokens: int = 0,
    top_p: float = 0.7,
    repetition_penalty: float = 1.2,
    temperature: float = 0.7,
    compile: bool = False,
    iterative_prompt: bool = True,
    max_length: int = 2048,
    chunk_length: int = 150,
    prompt_text=None,
    prompt_tokens=None,
    # plus whatever other args original code has...
):
    """
    A wrapper around your original generate_long that ensures we don't exceed max_seq_len,
    and monkey-patches 'generate' inside so we call our patched_generate.
    """
    from fish_speech.models.text2semantic.inference import generate_long as original_generate_long_internal
    from loguru import logger

    if max_length > model.config.max_seq_len:
        logger.warning(f"[PATCHED] 'max_length'={max_length} > model.config.max_seq_len={model.config.max_seq_len}."
                       " Forcing max_length = model.config.max_seq_len to avoid mismatch.")
        max_length = model.config.max_seq_len

    # >>> We monkey-patch generate in this scope so calls inside generate_long will use patched_generate
    from fish_speech.models.text2semantic import inference as inf_mod
    inf_mod.generate = patched_generate

    return original_generate_long_internal(
        model=model,
        device=device,
        decode_one_token=decode_one_token,
        text=text,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        compile=compile,
        iterative_prompt=iterative_prompt,
        max_length=max_length,
        chunk_length=chunk_length,
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens,
    )

# We override the name used by Text2SemanticWrapper
generate_long = patched_generate_long


class Text2SemanticWrapper:
    """
    Wraps your text2semantic model.
    *Compiles once* in __init__ if do_compile=True,
    then reuses that compilation for all segments.
    Also uses our patched generate_long to avoid dimension mismatch.
    """
    def __init__(self, checkpoint_path: str, device: str, use_half: bool, do_compile: bool):
        precision = torch.float16 if use_half else torch.bfloat16
        from fish_speech.models.text2semantic.inference import load_model as text2sem_load
        self.model, self.decode_one_token = text2sem_load(
            checkpoint_path=checkpoint_path,
            device=device,
            precision=precision,
            compile=do_compile,
            is_agent=False
        )
        self.model.to(device)
        with torch.device(device):
            self.model.setup_caches(
                max_batch_size=1,
                max_seq_len=self.model.config.max_seq_len,
                dtype=next(self.model.parameters()).dtype,
            )
        self.device = device
        self._compiled_once = do_compile
        logger.info(f"[Text2Semantic] Model loaded. do_compile={do_compile}")

        if do_compile:
            self._run_dummy_generation()

    def _run_dummy_generation(self):
        logger.info("[Text2Semantic] Doing a dummy generation to prime compilation...")
        _ = generate_long(
            model=self.model,
            device=self.device,
            decode_one_token=self.decode_one_token,
            text="Hello",
            num_samples=1,
            max_new_tokens=0,
            top_p=0.7,
            repetition_penalty=1.2,
            temperature=0.7,
            compile=False,
        )
        logger.info("[Text2Semantic] Dummy gen finished. Model compiled once.")

    def generate_codes(
        self,
        text: str,
        prompt_text: str,
        prompt_tokens_file: str = "fake.pt",
        num_samples: int = 2
    ) -> str:
        import torch

        max_new_tokens = 0
        top_p = 0.7
        repetition_penalty = 1.2
        temperature = 0.7
        chunk_length = 100
        iterative_prompt = True

        if os.path.exists(prompt_tokens_file):
            prompt_tokens = [torch.load(prompt_tokens_file).to(self.device)]
            prompt_texts = [prompt_text]
        else:
            logger.warning(f"No prompt tokens found: {prompt_tokens_file}")
            prompt_tokens = None
            prompt_texts = None

        generator = generate_long(
            model=self.model,
            device=self.device,
            decode_one_token=self.decode_one_token,
            text=text,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            compile=False,
            iterative_prompt=iterative_prompt,
            max_length=self.model.config.max_seq_len,  # dimension fix
            chunk_length=chunk_length,
            prompt_text=prompt_texts,
            prompt_tokens=prompt_tokens,
        )

        sample_idx = 0
        current_codes = []
        output_files = []
        for response in generator:
            if response.action == "sample":
                current_codes.append(response.codes)
            elif response.action == "next":
                if current_codes:
                    out_name = f"codes_{sample_idx}.pt"
                    combined = torch.cat(current_codes, dim=1)
                    torch.save(combined.cpu(), out_name)
                    logger.info(f"[Text2Semantic] Saved {out_name}")
                    output_files.append(out_name)
                    current_codes = []
                    sample_idx += 1
            else:
                logger.error(f"[Text2Semantic] Unexpected response: {response}")

        if output_files:
            return output_files[-1]
        return ""


# ---------------------------------------------------------------
#  >>> TTSService: integrates (1) VQGAN and (2) Text2Semantic <<<
# ---------------------------------------------------------------
class TTSService:
    """
    3-step pipeline:
      1) VQGAN encode -> "fake.pt"
      2) Text2Semantic -> "codes_1.pt"
      3) VQGAN decode -> final WAV
    """
    def __init__(
        self,
        text2semantic_checkpoint: str,
        vqgan_checkpoint_path: str,
        vqgan_config_name: str = "firefly_gan_vq",
        device: str = "cuda",
        half_text2semantic: bool = False,
        compile_text2semantic: bool = True
    ):
        self.device = device
        logger.info("Loading VQGAN model once ...")
        self.vqgan_model = load_vqgan_model(
            config_name=vqgan_config_name,
            checkpoint_path=vqgan_checkpoint_path,
            device=device
        )
        logger.info("Loading Text2Semantic model once ...")

        self.text2semantic = Text2SemanticWrapper(
            checkpoint_path=text2semantic_checkpoint,
            device=device,
            use_half=half_text2semantic,
            do_compile=compile_text2semantic
        )
        logger.info("[TTSService] All models ready.")

    def run_tts_pipeline(
        self,
        input_wav: str,
        french_text: str,
        english_prompt_text: str,
        out_wav: str,
        fake_code: str = "fake.pt",
        codes_out: str = "codes_1.pt"
    ) -> float:
        # Step 1: VQGAN encode
        vqgan_encode_audio(self.vqgan_model, input_wav, fake_code, device=self.device)

        # Step 2: text2semantic
        codes_file = self.text2semantic.generate_codes(
            text=french_text,
            prompt_text=english_prompt_text,
            prompt_tokens_file=fake_code,
            num_samples=2
        )
        if not codes_file:
            logger.error("[TTSService] No codes generated, skipping.")
            return 0.0

        if codes_file != codes_out:
            os.replace(codes_file, codes_out)

        # Step 3: VQGAN decode
        vqgan_decode_codes(self.vqgan_model, codes_out, out_wav, device=self.device)
        duration = sf.info(out_wav).duration
        logger.info(f"[TTSService] Created {out_wav} with duration={duration:.2f}s")
        return duration


# ---------------------------------------------------------------
#  >>> MAIN SCRIPT
# ---------------------------------------------------------------
if __name__ == "__main__":
    json_path = "/home/amine/DeepDub/Data/merged_all_films.json"
    output_json_path = "/home/amine/DeepDub/Data/merged_all_films_with_tts.json"
    fish_folder = "/home/amine/DeepDub/Data/fish"
    os.makedirs(fish_folder, exist_ok=True)

    text2semantic_ckpt = "/home/amine/DeepDub/fisher/checkpoints/fish-speech-1.5-yth-lora"
    vqgan_ckpt = "/home/amine/DeepDub/fisher/checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
    service = TTSService(
        text2semantic_checkpoint=text2semantic_ckpt,
        vqgan_checkpoint_path=vqgan_ckpt,
        vqgan_config_name="firefly_gan_vq",
        device="cuda",
        half_text2semantic=False,
        compile_text2semantic=True
    )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    base_dir = os.path.dirname(json_path)
    total_rows = len(data)
    logger.info(f"Found {total_rows} total rows in the JSON.")

    missing_indices = []
    for i in range(total_rows):
        out_wav = os.path.join(fish_folder, f"{i}.wav")
        if not os.path.exists(out_wav):
            missing_indices.append(i)

    logger.info(f"{len(missing_indices)} missing segments out of {total_rows} total.")

    for i in tqdm(missing_indices, desc="Processing missing rows"):
        seg = data[i]
        out_wav = os.path.join(fish_folder, f"{i}.wav")

        eng_audio_rel = seg["english"]["audio_paths"][0].lstrip("/")
        eng_audio_full = os.path.join(base_dir, eng_audio_rel)
        eng_text = seg["english"]["text"]
        fr_text = seg["french"]["text"]

        logger.info(f"--- Processing row index={i} ---")
        dur = service.run_tts_pipeline(
            input_wav=eng_audio_full,
            french_text=fr_text,
            english_prompt_text=eng_text,
            out_wav=out_wav,
            fake_code="fake.pt",
            codes_out="codes_1.pt"
        )

        seg.setdefault("tts", {})
        seg["tts"]["fish-tts"] = {
            "audio_path": f"/fish/{i}.wav",
            "duration": dur
        }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Done! JSON => {output_json_path}")