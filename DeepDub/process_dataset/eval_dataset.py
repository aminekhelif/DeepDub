import json
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

def main():
    # --------------------------------------------------
    # 1. Define file paths
    # --------------------------------------------------
    input_file = "./Data/Dataset.json"       # <-- Replace with your actual JSON filename
    output_file = "similarity_scores.json"   # <-- Output filename

    # --------------------------------------------------
    # 2. Load your JSON data
    # --------------------------------------------------
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # --------------------------------------------------
    # 3. Check for CUDA availability & load model on GPU if available
    # --------------------------------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model_name = "sentence-transformers/LaBSE"  # or another multilingual model
    model = SentenceTransformer(model_name, device=device)

    # --------------------------------------------------
    # 4. Process each segment & compute similarity (with progress bar)
    # --------------------------------------------------
    for segment in tqdm(data, desc="Processing segments"):
        english_text = segment["english"]["text"]
        french_text  = segment["french"]["text"]

        # Encode texts into embeddings on GPU (if available)
        emb_en = model.encode(english_text, convert_to_tensor=True, device=device)
        emb_fr = model.encode(french_text,  convert_to_tensor=True, device=device)

        # Calculate cosine similarity (stays on GPU if convert_to_tensor=True)
        similarity_score = util.cos_sim(emb_en, emb_fr).item()

        # Store the similarity in the segment
        segment["similarity_score"] = similarity_score

    # --------------------------------------------------
    # 5. Save updated data (with similarity) to a new JSON file
    # --------------------------------------------------
    with open(output_file, 'w', encoding='utf-8') as out_f:
        json.dump(data, out_f, ensure_ascii=False, indent=4)

    print(f"Done! Similarity scores saved to '{output_file}'.")

if __name__ == "__main__":
    main()