from flask import Flask, request, render_template, jsonify
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import spacy
import re
import os

# Initialize Flask app
app = Flask(__name__)

# Load models
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
biobert_tokenizer = AutoTokenizer.from_pretrained("deepset/bert-large-uncased-whole-word-masking-squad2")
biobert_model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-large-uncased-whole-word-masking-squad2", torch_dtype=torch.float32)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")
# Load dataset
df = pd.read_csv("medDataset_processed.csv", on_bad_lines="skip")
df = df.dropna()
df["cleaned_text"] = df["Question"].apply(lambda x: re.sub(r"\W+", " ", x.lower()))
stored_questions = df["cleaned_text"].tolist()
stored_answers = df["Answer"].tolist()
stored_embeddings = sbert_model.encode(stored_questions, convert_to_tensor=True)

# Function to get the best match
def get_best_match(query):
    query_embedding = sbert_model.encode(query, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(query_embedding, stored_embeddings)
    best_match_idx = torch.argmax(cosine_scores).item()
    return stored_answers[best_match_idx]

# Flask routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_query = request.json.get("query", "")
    response = get_best_match(user_query)
    return jsonify({"response": response})

# Vercel handler function
def handler(request):
    with app.app_context():
        return app.full_dispatch_request()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use 10000 as default for Vercel compatibility
    app.run(host="0.0.0.0", port=port, debug=True)
