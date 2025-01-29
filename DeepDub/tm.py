import os
import json
import yaml
from typing import List, Optional
from pydantic import BaseModel
from DeepDub.logger import logger

from openai import OpenAI
import argparse

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')

def _load_config(config_path: str) -> dict:
    """Loads YAML config from disk (including openai_api_key)."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

_cfg = _load_config(CONFIG_PATH)
_openai_api_key = _cfg.get("openai_api_key")
_base_url = _cfg.get("base_url", 'https://api.openai.com/v1/')
if not _openai_api_key:
    raise ValueError("No 'openai_api_key' found in config.yaml.")

_client = OpenAI(base_url=_base_url, api_key=_openai_api_key)
logger.info("OpenAI client initialized from config.yaml.")

class IndexedTranslationItem(BaseModel):
    """
    The LLM must return exactly:
      {
        "index": <int>,
        "text": <string>,
        "translated_text": <string>
      }

    We'll only use 'index' to map back and 'translated_text' to store in final output.
    The LLM's 'text' is just for alignment checking (to see that it hasn't dropped or reordered).
    """

    index: int
    text: str
    translated_text: str


class IndexedTranslationList(BaseModel):
    """
    The final LLM output is a JSON array, each item -> IndexedTranslationItem.
    We expect the length to match the input length exactly.
    """
    items: List[IndexedTranslationItem]

    def to_list(self) -> List[IndexedTranslationItem]:
        return self.items

class DeepDubTranslator:
    """
    This translator does the following:
      1) Pre-process the 'segments' to ensure each has a unique 'index' and a valid 'text'.
      2) Send them in chunks to the LLM, expecting an equal-length JSON array of items:
         [ { "index": i, "text": "...", "translated_text": "..." }, ... ]
      3) On parse or alignment failures, subdivide chunk or fall back.
      4) Merge 'translated_text' back into final segments by matching the 'index'.
      5) Write final JSON with the new 'translated_text' field.
    """

    def __init__(self, client: OpenAI, default_model: str = "gpt-4o", max_retries: int = 2):
        self._client = client
        self._default_model = default_model
        self.max_retries = max_retries
        logger.info("DeepDubTranslator using index approach. model='%s' max_retries=%d",
                    default_model, max_retries)

    def translate_json(
        self,
        segments: List[dict],
        diar_json_path: str,
        target_language: str = "French",
        model_name: Optional[str] = None,
        chunk_size: int =40
    ) -> str:
        """
        Main entry point:
          1) Enforce each segment has {index, text}:
             - If 'index' is missing, assign a unique integer.
          2) Chunk them up to 'chunk_size'.
          3) For each chunk, call LLM to get translations.
          4) Merge into final list -> each has original fields + 'translated_text'.
          5) Save final to 'diarization_translated.json' in same folder as diar_json_path.
          6) Return that path.

        The final output does NOT keep 'index' or any LLM-changed numeric fields.
        """
        if model_name is None:
            model_name = self._default_model

        self._assign_indexes_if_missing(segments)

        n = len(segments)
        logger.info("Starting indexed translation: model='%s', language='%s', total_segments=%d",
                    model_name, target_language, n)

        base_dir = os.path.dirname(diar_json_path)
        output_file = os.path.join(base_dir, "diarization_translated.json")

        final_segments = [dict(seg) for seg in segments]

        idx = 0
        chunk_id = 1
        while idx < n:
            sub_data = final_segments[idx : idx + chunk_size]
            logger.info("Translating chunk %d containing %d items...", chunk_id, len(sub_data))
            chunk_id += 1

            partial_result = self._translate_chunk_subdivide(sub_data, target_language, model_name)
            for i, item in enumerate(partial_result):
                global_i = idx + i
                final_segments[global_i]["translated_text"] = item.translated_text

            idx += chunk_size

        for seg in final_segments:
            if "index" in seg:
                del seg["index"]

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_segments, f, indent=4, ensure_ascii=False)

        logger.info("Translation complete. Wrote => %s", output_file)
        return output_file

    ########################################################
    #  Pre-processing: Assign indexes if needed
    ########################################################

    def _assign_indexes_if_missing(self, segments: List[dict]):
        """
        Make sure each segment has a unique integer 'index'.
        If any are missing or invalid, assign 0..n-1 in order.
        """
        for i, seg in enumerate(segments):
            if "index" not in seg or not isinstance(seg["index"], int) or seg["index"] < 0:
                seg["index"] = i
            if "text" not in seg or not isinstance(seg["text"], str):
                seg["text"] = ""

    ########################################################
    #  Subdividing approach with fallback
    ########################################################

    def _translate_chunk_subdivide(
        self,
        data: List[dict],
        target_language: str,
        model_name: str
    ) -> List[IndexedTranslationItem]:
        """
        Translate a chunk of data => same length. If merges/drops => retry or subdivide.
        ALSO: If any 'translated_text' is empty, treat it like a parse failure => retry or subdivide.
        """
        size = len(data)
        if size == 0:
            return []

        if size == 1:
            return self._translate_single(data[0], target_language, model_name)

        for attempt in range(self.max_retries + 1):
            parsed_list = self._call_llm_parse(data, target_language, model_name)
            if not parsed_list:
                logger.warning("Chunk parse error for size=%d attempt=%d => fallback or subdiv",
                               size, attempt + 1)
                if attempt == self.max_retries:
                    return self._subdivide_and_combine(data, target_language, model_name)
                else:
                    continue

            items = parsed_list.items

            # 1) Check length
            if len(items) != size:
                logger.warning("LLM merged/dropped => got=%d, expected=%d attempt=%d => subdiv",
                               len(items), size, attempt+1)
                if attempt == self.max_retries:
                    return self._subdivide_and_combine(data, target_language, model_name)
                continue

            # 2) Check indexes
            if not self._validate_indexes(data, items):
                logger.warning("LLM re-ordered or changed indexes. attempt=%d => subdiv",
                               attempt+1)
                if attempt == self.max_retries:
                    return self._subdivide_and_combine(data, target_language, model_name)
                continue

            # 3) Check for empty translated_text
            if any(not (itm.translated_text and itm.translated_text.strip()) for itm in items):
                logger.warning("LLM returned at least one empty 'translated_text'. attempt=%d => subdiv",
                               attempt+1)
                if attempt == self.max_retries:
                    return self._subdivide_and_combine(data, target_language, model_name)
                else:
                    continue

            # If all checks passed, return items
            return items

        # If all attempts fail, fallback
        return self._fallback_fill(data)

    def _subdivide_and_combine(
        self,
        data: List[dict],
        target_language: str,
        model_name: str
    ) -> List[IndexedTranslationItem]:
        """
        Subdivide a chunk into two halves, translate each half,
        and combine the results. This helps if the LLM merges/drops items or returns empties.
        """
        size = len(data)
        if size <= 1:
            # Edge case: just translate single
            return self._translate_single(data[0], target_language, model_name)

        mid = size // 2
        left = data[:mid]
        right = data[mid:]
        logger.info("Subdivide chunk => left=%d items, right=%d items", len(left), len(right))

        L = self._translate_chunk_subdivide(left, target_language, model_name)
        R = self._translate_chunk_subdivide(right, target_language, model_name)
        return L + R

    def _translate_single(
        self,
        seg: dict,
        target_language: str,
        model_name: str
    ) -> List[IndexedTranslationItem]:
        """
        Try translating a single item multiple times. If it fails or returns empty, fallback.
        """
        for attempt in range(self.max_retries + 1):
            res_list = self._call_llm_parse([seg], target_language, model_name)
            if res_list and len(res_list.items) == 1:
                item = res_list.items[0]
                # Check index match
                if self._validate_indexes([seg], [item]):
                    # Check for non-empty translation
                    if item.translated_text and item.translated_text.strip():
                        return res_list.items

            logger.warning("Single item parse fail or empty text attempt=%d => fallback if last.",
                           attempt+1)
            if attempt == self.max_retries:
                idx_val = seg.get("index", -1)
                txt_val = seg.get("text", "")
                return [IndexedTranslationItem(index=idx_val,
                                               text=txt_val,
                                               translated_text="")]

        # If somehow it keeps failing, fallback
        idx_val = seg.get("index", -1)
        txt_val = seg.get("text", "")
        return [IndexedTranslationItem(index=idx_val, text=txt_val, translated_text="")]

    def _fallback_fill(self, data: List[dict]) -> List[IndexedTranslationItem]:
        """
        Fallback: return items with empty translated_text if everything else fails.
        """
        logger.warning("Fallback fill => size=%d => 'Unknown' or empty translations", len(data))
        items = []
        for seg in data:
            idx_val = seg.get("index", -1)
            txt_val = seg.get("text", "")
            items.append(IndexedTranslationItem(index=idx_val,
                                                text=txt_val,
                                                translated_text=""))
        return items

    ########################################################
    # LLM call => (index, text, translated_text)
    ########################################################

    def _call_llm_parse(
        self,
        data: List[dict],
        target_language: str,
        model_name: str
    ) -> Optional[IndexedTranslationList]:
        """
        Instruct the LLM to produce a JSON array of the same length,
        each item => { "index", "text", "translated_text" }.
        We parse it with pydantic to ensure correctness.
        """
        system_prompt = (
            "You are a highly skilled translator. "
            "We have a list of segments, each with an 'index' (integer) and 'text' (string). "
            "First, read the entire list to fully understand the context. "
            "For each segment:\n"
            "- Keep the 'index' unchanged.\n"
            "- Keep the 'text' **exactly** as given (do not modify the source text). \n"
            "- Create or fill a 'translated_text' field with your translation into the target language.\n\n"

            "Important:\n"
            "- Preserve context, nuances, cultural references, wordplay, idiomatic expressions, acronyms, slang, etc.\n"
            "- Do **not** translate any names of people, companies, or brands.\n"
            "- All symbols, measurement units, or numbers must be spelled out in words in the target language "
            "(e.g., '5 km' -> 'cinq kilomètres', '%' -> 'pour cent', '$' -> 'dollars').\n"
            "- Try to keep the translation length close to the original text length (the output will be used for audio synthesis).\n"
            "- **Do not** remove, merge, or reorder segments. The final output must have the exact same number of items, "
            "with the same 'index' values in the same order.\n"
            "- Return a valid JSON array of objects. Each object must have the fields:\n"
            "    {\n"
            "      \"index\": <same integer>,\n"
            "      \"text\": <exact same text>,\n"
            "      \"translated_text\": <your translation>\n"
            "    }"
        )
        user_prompt = self._build_indexed_prompt(data, target_language)

        try:
            completion = self._client.beta.chat.completions.parse(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format=IndexedTranslationList
            )
            choice = completion.choices[0].message
            if choice.refusal:
                logger.error("LLM refused the request. refusal msg: %s", choice.refusal)
                return None

            chunk_obj = choice.parsed
            if not chunk_obj or not isinstance(chunk_obj, IndexedTranslationList):
                logger.error("LLM output is not an IndexedTranslationList.")
                return None

            return chunk_obj

        except Exception as e:
            logger.error("Error calling LLM parse: %s", e)
            return None

    def _build_indexed_prompt(self, data: List[dict], target_language: str) -> str:
        """
        Build the user-level prompt that enumerates each segment's index => text,
        asking the model to produce the final JSON array with 'index', 'text', and 'translated_text'.
        """

        lines = []
        for seg in data:
            idx_val = seg.get("index", -1)
            txt_val = seg.get("text", "").replace("\n", " ")
            lines.append(f"{idx_val} => {txt_val}")

        prompt = (
            f"Below are {len(data)} segments. Please translate each segment's 'text' into {target_language}.\n"
            "Remember:\n"
            "- Keep the same 'index' integer.\n"
            "- Do **not** modify the 'text' field.\n"
            "- Add a 'translated_text' field with the appropriate translation.\n\n"
            "Return a **JSON array** of exactly the same length, each item in the form:\n"
            "{\n"
            "  \"index\": <same integer>,\n"
            "  \"text\": <same original text>,\n"
            "  \"translated_text\": <your translation>\n"
            "}\n\n"
            "Here are the segments (index => text):\n"
        )
        prompt += "\n".join(lines)
        return prompt

    def _validate_indexes(self, original_data: List[dict], llm_items: List[IndexedTranslationItem]) -> bool:
        """
        Ensure the LLM's returned indexes match one-to-one 
        with our original list order. If there's a mismatch, consider it a parse failure.
        """
        for orig, llm_item in zip(original_data, llm_items):
            if orig["index"] != llm_item.index:
                logger.warning("Mismatch index: original=%d vs llm=%d, text=%s",
                               orig["index"], llm_item.index, llm_item.text)
                return False
        return True

############################################################
#  GLOBAL TRANSLATOR INSTANCE
############################################################

translator = DeepDubTranslator(
    client=_client,
    default_model="gpt-4o",
    max_retries=2
)
logger.info("Global 'translator' (index-based) is ready.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI for segment-based translation.")
    parser.add_argument("--input", required=True,
                        help="Path to a JSON containing segments with optional index/text.")
    parser.add_argument("--output", help="Path to diar_simple.json (for directory reference).",
                        default=None)
    parser.add_argument("--target", help="Target language", default="French")
    parser.add_argument("--chunk-size", type=int, default=40)
    parser.add_argument("--model", default="gpt-4", help="Model name.")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as fin:
        segments_data = json.load(fin)

    if not isinstance(segments_data, list):
        raise ValueError(f"Input JSON must be a list, got {type(segments_data)}")

    if not args.output:
        diar_json_path = os.path.abspath(args.input)
    else:
        diar_json_path = os.path.abspath(args.output)

    final_path = translator.translate_json(
        segments=segments_data,
        diar_json_path=diar_json_path,
        target_language=args.target,
        model_name=args.model,
        chunk_size=args.chunk_size
    )

    print(f"CLI translation done. Output => {final_path}")