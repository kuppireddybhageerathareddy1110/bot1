# from flask import Flask, request, render_template, jsonify
# import pandas as pd
# import torch
# from sentence_transformers import SentenceTransformer, util
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering
# from gensim.models import Word2Vec
# import nltk
# import spacy
# import re
# import string
<<<<<<< HEAD

# app = Flask(__name__)

# # Load models
# sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
# biobert_tokenizer = AutoTokenizer.from_pretrained("deepset/bert-large-uncased-whole-word-masking-squad2")
# biobert_model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-large-uncased-whole-word-masking-squad2", torch_dtype=torch.float32)
# nlp = spacy.load("en_core_web_sm")

# # Load dataset
# df = pd.read_csv("medDataset_processed.csv", on_bad_lines="skip")
# df = df.dropna()
# df["cleaned_text"] = df["Question"].apply(lambda x: re.sub(r"\W+", " ", x.lower()))
# stored_questions = df["cleaned_text"].tolist()
# stored_answers = df["Answer"].tolist()
# stored_embeddings = sbert_model.encode(stored_questions, convert_to_tensor=True)

# def get_best_match(query):
#     query_embedding = sbert_model.encode(query, convert_to_tensor=True)
#     cosine_scores = util.pytorch_cos_sim(query_embedding, stored_embeddings)
#     best_match_idx = torch.argmax(cosine_scores).item()
#     return stored_answers[best_match_idx]

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/chat", methods=["POST"])
# def chat():
#     user_query = request.json.get("query", "")
#     response = get_best_match(user_query)
#     return jsonify({"response": response})

# if __name__ == "__main__":
#     app.run(debug=True)
# from flask import Flask, request, render_template, jsonify
# import pandas as pd
# import torch
# from sentence_transformers import SentenceTransformer, util
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering
# from gensim.models import Word2Vec
# import nltk
# import spacy
# import re
# import string
=======
>>>>>>> 949f8d36ac44403c6451148ef07ebd6b89ace1b5
# import os
# import spacy
# import subprocess

# app = Flask(__name__)




# # Load models
# sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
# biobert_tokenizer = AutoTokenizer.from_pretrained("deepset/bert-large-uncased-whole-word-masking-squad2")
# biobert_model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-large-uncased-whole-word-masking-squad2", torch_dtype=torch.float32)
# # nlp = spacy.load("en_core_web_sm")
# try:
#     nlp = spacy.load("en_core_web_sm")
# except OSError:
#     subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
#     nlp = spacy.load("en_core_web_sm")
# # Load dataset
# df = pd.read_csv("medDataset_processed.csv", on_bad_lines="skip")
# df = df.dropna()
# df["cleaned_text"] = df["Question"].apply(lambda x: re.sub(r"\W+", " ", x.lower()))
# stored_questions = df["cleaned_text"].tolist()
# stored_answers = df["Answer"].tolist()
# stored_embeddings = sbert_model.encode(stored_questions, convert_to_tensor=True)

# def get_best_match(query):
#     query_embedding = sbert_model.encode(query, convert_to_tensor=True)
#     cosine_scores = util.pytorch_cos_sim(query_embedding, stored_embeddings)
#     best_match_idx = torch.argmax(cosine_scores).item()
#     return stored_answers[best_match_idx]

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/chat", methods=["POST"])
# def chat():
#     user_query = request.json.get("query", "")
#     response = get_best_match(user_query)
#     return jsonify({"response": response})

# # if __name__ == "__main__":
# #     app.run(debug=True)



<<<<<<< HEAD

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))  # Default to 5000 if no PORT is found
#     app.run(host="0.0.0.0", port=port)

=======
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 10000))  # Default to 10000 if no PORT is found
#     app.run(host="0.0.0.0", port=port, debug=True)
>>>>>>> 949f8d36ac44403c6451148ef07ebd6b89ace1b5
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render default is 10000
    app.run(host="0.0.0.0", port=port, debug=True)
