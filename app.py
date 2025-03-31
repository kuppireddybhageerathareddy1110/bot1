from flask import Flask, request, render_template, jsonify
import os
import torch
import spacy
import re
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

app = Flask(__name__)

# Lazy-load models only when required
def load_sbert():
    return SentenceTransformer("all-MiniLM-L6-v2")

def load_biobert():
    tokenizer = AutoTokenizer.from_pretrained("deepset/bert-large-uncased-whole-word-masking-squad2")
    model = AutoModelForQuestionAnswering.from_pretrained(
        "deepset/bert-large-uncased-whole-word-masking-squad2", torch_dtype=torch.float32
    )
    return tokenizer, model

def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_query = request.json.get("query", "")
    
    # Load SBERT only when needed
    sbert_model = load_sbert()
    
    # Dummy response (for now)
    response = f"Processed: {user_query}"
    
    return jsonify({"response": response})

# For Vercel compatibility, this function should be at the bottom.
def handler(request):
    with app.app_context():
        return app.full_dispatch_request()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Vercel environment port handling
    app.run(host="0.0.0.0", port=port, debug=True)
