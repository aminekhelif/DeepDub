import json
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

def main():

    input_file = "./Data/Dataset.json"       
    output_file = "similarity_scores.json"   


    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model_name = "sentence-transformers/LaBSE" 
    model = SentenceTransformer(model_name, device=device)


    for segment in tqdm(data, desc="Processing segments"):
        english_text = segment["english"]["text"]
        french_text  = segment["french"]["text"]

        emb_en = model.encode(english_text, convert_to_tensor=True, device=device)
        emb_fr = model.encode(french_text,  convert_to_tensor=True, device=device)

        similarity_score = util.cos_sim(emb_en, emb_fr).item()

        segment["similarity_score"] = similarity_score


    with open(output_file, 'w', encoding='utf-8') as out_f:
        json.dump(data, out_f, ensure_ascii=False, indent=4)

    print(f"Done! Similarity scores saved to '{output_file}'.")

if __name__ == "__main__":
    main()