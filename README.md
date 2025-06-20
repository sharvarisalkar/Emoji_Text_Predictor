# Emoji Text Predictor using S-BERT

Our project is a smart and interactive web app that predicts emojis for any sentence you type. We used Sentence-BERT (SBERT) to understand the meaning of the input text and match it with the most relevant emojis. Instead of using heavy models, our system gives fast results using semantic similarity with a custom emoji keyword list.

**📖 Project Overview -**
The Emoji Text Predictor is an NLP-based tool that takes any sentence or phrase as input and returns emojis that match the semantic meaning of the sentence. Instead of relying on exact keyword matching or heavy sequence-based models, this system uses SBERT embeddings and cosine similarity to find the most contextually relevant emojis. The model filters out irrelevant words like stopwords and focuses on meaningful keywords in the text.
It is implemented as a Flask web application, allowing users to interact with the tool in real-time through a simple UI.

**⚙️ Methodology -**
1. User Input: The user enters a sentence or phrase through the web interface.
2. Text Preprocessing: The input is cleaned (lowercasing, stopword removal, punctuation removal, lemmatization).
3. Sentence Embedding with SBERT: The cleaned input is converted into a dense semantic vector using SBERT.
4. Loading Pre-Embedded Emoji Dataset: The system loads a keyword-emoji dataset (.py) where each keyword is already embedded using SBERT.
5. Cosine Similarity Matching: The user’s input vector is compared with the emoji dataset to find the closest matches.
6. Emoji Retrieval and Output: The most relevant emojis are selected and returned to the user.
7. Live Adaptability: The emoji-keyword list can be updated dynamically via a Python file without retraining the model.

**🛠️ Technologies Used -**
1. Python
2. Flask
3. Sentence-BERT (SBERT)
4. Cosine Similarity
5. HTML/CSS (for web interface)

**💡 Key Features -**
- Real-time emoji prediction from user input
- SBERT-based semantic understanding (no model training required)
- Flask-based web interface
- Lightweight and scalable for real-time use

