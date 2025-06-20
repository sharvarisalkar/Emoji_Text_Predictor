from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
import numpy as np
from emoji_data import emojis, emo_list

app = Flask(__name__)

# Load SBERT model and vectors
print("Loading Sentence-BERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Sentence-BERT model loaded successfully.")

# Load precomputed vectors
doc_vectors = np.load('./saved_models/sbert_vectors.npy')

# Preprocess emoji keywords
e_l = [str(i.replace("_", " ")).lower() for i in emo_list]
emoji_mapping = {k.lower().strip(':').replace('_', ' '): v for k, v in emojis.items()}

# Define common words to exclude (but keep gender words)
common_words = {"the", "is", "and", "a", "an", "to", "in", "on", "for", "with", "of"}

def find_best_matching_emoji(word):
    """Finds the best matching emoji for a given word using cosine similarity."""
    text_vector = model.encode([word])[0]

    # Compute cosine similarity with stored emoji vectors
    similarities = np.dot(doc_vectors, text_vector) / (np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(text_vector))

    # Get the best matching emoji
    top_index = np.argmax(similarities)
    emoji_key = e_l[top_index]

    return emoji_mapping.get(emoji_key, "")

def text_to_emoji(text):
    """Converts meaningful words in text to emojis while ignoring common words."""
    words = text.lower().split()
    emoji_result = []

    for word in words:
        if word in common_words:
            emoji_result.append("")  # Ignore common words
        else:
            emoji = find_best_matching_emoji(word)
            emoji_result.append(emoji)

    return ' '.join(filter(None, emoji_result))  # Remove empty spaces

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        input_text = request.form['input_text']
        emoji_result = text_to_emoji(input_text)
        return render_template('index.html', result=emoji_result, original_text=input_text)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)  # Explicitly disabling watchdog to prevent extra reloading
