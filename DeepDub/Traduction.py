import os
import json
import getpass
from typing import List, Optional

import openai
from pydantic import BaseModel, ValidationError, RootModel

# ========= CONFIGURATION ==========

openai_api_key = getpass.getpass("Veuillez saisir votre clé OpenAI : ")
openai.api_key = openai_api_key  # Clé OpenAI

model_name = "gpt-3.5-turbo"
target_language = "French"
input_file_path = "input_text.json"
output_file_path = "output_text.json"
CHUNK_SIZE = 30

# ========= PYDANTIC MODELS ==========

class TranslatedSegment(BaseModel):
    
    start: Optional[float]
    end: Optional[float]
    text: str
    speaker: Optional[str]
    translated_text: str

class TranslatedSegmentList(RootModel[List[TranslatedSegment]]):
    """A root model representing a list of TranslatedSegment objects."""

# ========= FONCTIONS ==========

def build_user_message_for_chunk(chunk: List[dict]) -> str:
    prompt_intro = (
        "Below is a list of segments in JSON format. "
        "For each segment:\n"
        "- Do NOT modify the 'text' field (it must remain in the source language).\n"
        f"- Create or fill 'translated_text' with the translation into {target_language}.\n\n"
        "Return a valid JSON array of objects, where each object preserves the original fields "
        "(segment, start, end, text) and includes a new field 'translated_text'.\n"
        "Example:\n"
        "[\n"
        "  {\n"
          
        "    \"start\": 0.0,\n"
        "    \"end\": 5.0,\n"
        "    \"text\": \"This is the original text.\",\n"
        "    \"speaker\": \"SPEAKER_1\",\n"
        "    \"translated_text\": \"Voici le texte traduit.\" \n"
        "  }\n"
        "]\n\n"
        "Important:\n"
        "- Translate from detected source language to the target language.\n"
        "- Preserve nuances, cultural references, wordplay, idiomatic expressions, etc.\n"
        "- Any symbols, measurement units, or numbers must be written out in words.\n"
        
    )

    chunk_json = json.dumps(chunk, ensure_ascii=False, indent=2)
    user_message = f"{prompt_intro}\n\nHere is the chunk to translate:\n{chunk_json}\n\n"
    return user_message

def translate_chunk(chunk: List[dict]) -> List[TranslatedSegment]:
    user_message = build_user_message_for_chunk(chunk)

    system_prompt = (
        "You are a highly skilled translator. "
        "First, read the entire set of segments to fully grasp overall context. "
        "Then, translate each segment's 'text' field into the target language while preserving context, "
        "nuances, cultural references, wordplay, idiomatic expressions, acronyms, measurement units, slang, "
        "and so on. If there's any wordplay or idiomatic expression that needs adaptation, keep the same intent and style. "
        "make sure to not translate the names of people, companies, or brands. "
        "All symbols, numbers, or measurement units must be written out in full words in the target language. (e.g., 5 kilometers -> cinq kilomètres, % -> pour cent, $ -> dollars, etc.) "
    )

    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2
        )
        llm_output = response["choices"][0]["message"]["content"].strip()

        # On essaye de parser la réponse comme JSON
        data = json.loads(llm_output)
        # On le valide via Pydantic RootModel
        validated = TranslatedSegmentList.parse_obj(data)
        return validated.root
    except (json.JSONDecodeError, ValidationError) as e:
        print("Erreur de parsing ou de validation Pydantic :", e)
        return []
    except Exception as e:
        print(f"Erreur d'appel OpenAI pour le chunk : {e}")
        return []

def main():
    
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Le fichier d'entrée doit être une liste JSON de segments.")

    all_translated_segments = []

    for i in range(0, len(data), CHUNK_SIZE):
        chunk = data[i : i + CHUNK_SIZE]
        print(f"\n--- Traitement du chunk n°{i//CHUNK_SIZE + 1} contenant {len(chunk)} segments ---")

        translated_items = translate_chunk(chunk)
        # Ajout au tableau global
        all_translated_segments.extend([item.dict() for item in translated_items])

    # Sauvegarde
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(all_translated_segments, f, indent=4, ensure_ascii=False)

    print(f"\nTraduction terminée ! Résultats enregistrés dans : {output_file_path}")

if __name__ == "__main__":
    main()
