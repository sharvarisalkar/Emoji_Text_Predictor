import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from emoji_data import emojis, emo_list
import os

# Load SBERT model
print("Loading Sentence-BERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Sentence-BERT model loaded successfully.")

# Preprocess emoji keywords
e_l = [str(i.replace("_", " ")).lower() for i in emo_list]
emoji_mapping = {k.lower().strip(':').replace('_', ' '): v for k, v in emojis.items()}

# Encode emoji keywords into SBERT vectors
print("Encoding emoji keywords...")
doc_vectors = model.encode(e_l, show_progress_bar=True)

# Save the model and vectors
output_dir = './saved_models'
os.makedirs(output_dir, exist_ok=True)
model.save(f'{output_dir}/sbert_model')
np.save(f'{output_dir}/sbert_vectors.npy', doc_vectors)
print("Models and vectors saved successfully.")

# Function to get top emoji matches
def text_to_emoji(text, top_k=5):
    text_vector = model.encode([text])[0]

    # Compute cosine similarity
    similarities = np.dot(doc_vectors, text_vector) / (np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(text_vector))

    # Get top-k best matches
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_emojis = [emoji_mapping.get(e_l[i], "") for i in top_indices]

    return [emoji for emoji in top_emojis if emoji]  # Remove empty results

# Example usage
'''
example_text = "I'm feeling very happy today!"
result = text_to_emoji(example_text)
print(f"Input: {example_text}")
print(f"Top 5 emojis: {' '.join(result)}")
'''
