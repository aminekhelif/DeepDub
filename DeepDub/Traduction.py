import os
import openai
import json

# ====== CONFIGURATION ======
openai.api_key = "sk-..."                 # Your OpenAI API key
model_name = "gpt-3.5-turbo"            # Model to use for translation
target_language = "French"               # Change to your desired target language
input_file_path = "diar_simple.json"         # Input JSON file path
output_file_path = "example_translated.json"  # Output JSON file path
# ==========================

def translate_text(text, target_language):
    """
    Translates the given text into the target_language using OpenAI's ChatCompletion.
    The system instruction includes a directive for detecting and preserving nuances
    (idiomatic expressions, cultural references, etc.).
    """
    try:
        # We use ChatCompletion with a system message to instruct the model
        # on how we want the translation to be performed.
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a highly skilled translator. "
                        "Your task is to detect the source language automatically and translate it into "
                        f"{target_language} while preserving context, nuances, cultural references, wordplay, "
                        "idiomatic expressions, acronyms, measurement units, and slang terms. "
                        "If there's wordplay or idiomatic expressions that need adaptation, do so while keeping "
                        "the same intent and style."
                    )
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.2  # Lower temperature for more consistent translations
        )

        # Extract the translated text from the response
        translated_text = response["choices"][0]["message"]["content"].strip()
        return translated_text
    
    except Exception as e:
        print(f"Error during translation: {e}")
        return text  # Fallback to original text if there's an error

def main():
    # 1. Read the input JSON file
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 2. Iterate over each item in the JSON list and translate the "text" field
    for item in data:
        original_text = item.get("text", "")
        
        # Provide an option for manual translation
        print(f"\nOriginal text: {original_text}")
        #user_input = input(
            #"Press ENTER to auto-translate OR type/paste your manual translation here:\n> "
        #).strip()
        
        #if user_input:
            # If user provided a manual translation, use it
            #translated_text = user_input
        #else:
            # Otherwise, auto-translate using OpenAI
        translated_text = translate_text(original_text, target_language)
        
        # Store or update the item with the translated text
        item["translated_text"] = translated_text
    
    # 3. Save the updated data to a new JSON file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"\nTranslation complete! Results saved to: {output_file_path}")

if __name__ == "__main__":
    main()
