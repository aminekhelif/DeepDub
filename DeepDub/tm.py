import os
import json
import yaml
from typing import List
from pydantic import BaseModel, model_validator, ValidationError
from DeepDub.logger import logger

from openai import OpenAI

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')

# -------------------- 1) Read Config & Initialize Client -------------------- #

def _load_config(config_path: str) -> dict:
    """Load YAML config (including OpenAI API key) from the given path."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

_cfg = _load_config(CONFIG_PATH)
_openai_api_key = _cfg.get("openai_api_key", None)
if not _openai_api_key:
    raise ValueError("No 'openai_api_key' found in config. Please add it to config.yaml.")

# Initialize the OpenAI client ONCE at import time
_client = OpenAI(api_key=_openai_api_key)
logger.info("OpenAI client initialized with key from config.yaml.")

# -------------------- 2) Pydantic Models for Structured Output -------------------- #

class TranslatedSegment(BaseModel):
    """
    Represents a single segment with both original text and translated text.
    If a field is missing/empty, we replace with defaults:
      - start/end => 0.0
      - text/speaker/translated_text => "Unknown"
    """

    start: float
    end: float
    text: str
    speaker: str
    translated_text: str

    @model_validator(mode="before")
    @classmethod
    def fill_missing_fields(cls, values):
        if "start" not in values or values["start"] is None:
            values["start"] = 0.0
        if "end" not in values or values["end"] is None:
            values["end"] = 0.0

        for field_name in ["text", "speaker", "translated_text"]:
            if field_name not in values or not str(values[field_name]).strip():
                values[field_name] = "Unknown"
        return values


class TranslatedChunk(BaseModel):
    """
    A container representing a list of TranslatedSegment objects.
    The LLM must return a JSON array, which we parse into `items`.
    """
    items: List[TranslatedSegment]

    def to_list(self) -> List[TranslatedSegment]:
        return self.items

# -------------------- 3) The Translation Class -------------------- #

class DeepDubTranslator:
    """
    A class providing chunk-wise translation of final JSON segments.
    - Reuses a single, globally initialized OpenAI client.
    - Saves output relative to the input file path.
    """

    def __init__(self, client: OpenAI, default_model: str = "gpt-4o"):
        """
        Instantiate with an already-initialized OpenAI client and a default model name.
        """
        self._client = client
        self._default_model = default_model
        logger.info("DeepDubTranslator is ready with model '%s'.", self._default_model)

    def translate_json(
        self,
        segments: List[dict],
        diar_json_path: str,
        target_language: str = "French",
        model_name: str = None,
        chunk_size: int = 30
    ) -> str:
        """
        Translate the final JSON segments, saving them to a path relative to 'input_file_path'.
        Returns the path for pipeline continuity.
        """
        if model_name is None:
            model_name = self._default_model

        logger.info(
            "Starting translation with model '%s', language '%s'. Segment count: %d",
            model_name, target_language, len(segments)
        )

        # Build the system prompt
        system_prompt = (
            "You are a highly skilled translator. "
            "First, read the entire set of segments to fully grasp overall context. "
            "Then, translate each segment's 'text' field into the target language while preserving context, "
            "nuances, cultural references, wordplay, idiomatic expressions, acronyms, measurement units, slang, etc. "
            "Make sure NOT to translate the names of people, companies, or brands. "
            "Symbols, numbers, or measurement units must be written out in words in the target language "
            "(e.g., '5 km' -> 'cinq kilomètres', '%' -> 'pour cent', '$' -> 'dollars')."
        )

        # Create 'diarization_translated.json' next to diar_json_path
        base_dir = os.path.dirname(diar_json_path)
        output_file_path = os.path.join(base_dir, "diarization_translated.json")

        all_segments = []
        for i in range(0, len(segments), chunk_size):
            chunk_data = segments[i : i + chunk_size]
            logger.info("Translating chunk %d (size=%d)", (i // chunk_size) + 1, len(chunk_data))

            translated_items = self._translate_chunk(
                chunk_data,
                model_name=model_name,
                system_prompt=system_prompt,
                target_language=target_language
            )
            # Convert Pydantic models to dict for JSON
            all_segments.extend(item.dict() for item in translated_items)

        # Save final JSON
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(all_segments, f, indent=4, ensure_ascii=False)

        logger.info("Translation complete. File saved at: %s", output_file_path)
        return output_file_path

    # -------------------- Private Helper -------------------- #

    def _translate_chunk(
        self,
        chunk: List[dict],
        model_name: str,
        system_prompt: str,
        target_language: str
    ) -> List[TranslatedSegment]:
        """
        Translate a single chunk of segments using the structured output approach:
        self._client.beta.chat.completions.parse(...).
        Returns a list of TranslatedSegment models.
        """

        user_message = self._build_user_prompt(chunk, target_language)

        try:
            completion = self._client.beta.chat.completions.parse(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,
                response_format=TranslatedChunk,
            )

            choice = completion.choices[0].message
            if choice.refusal:
                logger.error("LLM refused to respond for this chunk: %s", choice.refusal)
                return []

            chunk_obj = choice.parsed
            if not chunk_obj or not isinstance(chunk_obj, TranslatedChunk):
                logger.error("Unexpected format from LLM. Skipping chunk.")
                return []

            return chunk_obj.to_list()

        except ValidationError as ve:
            logger.error("Validation error in chunk translation: %s", ve)
            return []
        except Exception as e:
            logger.error("OpenAI API error for chunk: %s", e)
            return []

    def _build_user_prompt(self, chunk: List[dict], target_language: str) -> str:
        """
        Construct the user-facing prompt for a chunk of segments.
        """
        prompt_intro = (
            "Below is a list of segments in JSON format. "
            "For each segment:\n"
            "- Do NOT modify the 'text' field (it must remain in the source language).\n"
            f"- Create or fill 'translated_text' with the translation into {target_language}.\n\n"
            "Return a valid JSON array of objects, where each object preserves the original fields "
            "(start, end, text, speaker) and includes a new field 'translated_text'.\n\n"
            "Important:\n"
            "- Translate from the source language to the target language.\n"
            "- Preserve nuances, cultural references, wordplay, idiomatic expressions, etc.\n"
            "- Any symbols, measurement units, or numbers must be written out in words.\n"
        )
        chunk_json = json.dumps(chunk, ensure_ascii=False, indent=2)
        return f"{prompt_intro}\nHere is the chunk to translate:\n{chunk_json}\n"

# -------------------- 4) Create the Shared Translator at Import -------------------- #

translator = DeepDubTranslator(client=_client, default_model="gpt-4o")
logger.info("Global 'translator' instance is ready for use.")