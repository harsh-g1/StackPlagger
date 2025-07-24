from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import warnings
import numpy as np
import re
from threading import Lock
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# === Globals ===
model = None
clf = None
model_lock = Lock()
MODEL_PATH = "CPU_human_ai_classifier.pkl"

ALLOWED_TAGS = {"python", "java"}  


def load_model():
    global model, clf
    with model_lock:
        if model is None or clf is None:
            print("Loading embedding model and classifier...")
            model = SentenceTransformer("all-mpnet-base-v2")  
            with open(MODEL_PATH, "rb") as f:
                clf = pickle.load(f)


# Stylometric feature extractor (example with 17 features)
def extract_stylometric_features(text):
    lines = text.splitlines()
    num_lines = len(lines)
    avg_line_length = np.mean([len(line) for line in lines]) if lines else 0

    features = [
        len(text),                                         # total characters
        avg_line_length,                                   # avg line length
        sum(c.isdigit() for c in text),                    # digit count
        sum(c.isalpha() for c in text),                    # letter count
        sum(1 for c in text if c in "!@#$%^&*()_+-=[]{}"), # symbol count
        text.count('def'),                                 # 'def' keyword
        text.count('class'),                               # 'class' keyword
        text.count('import'),                              # 'import' keyword
        text.count('return'),                              # 'return' keyword
        text.count('for'),                                 # 'for' keyword
        text.count('while'),                               # 'while' keyword
        text.count('if'),                                  # 'if' keyword
        text.count('else'),                                # 'else' keyword
        text.count('elif'),                                # 'elif' keyword
        text.count('try'),                                 # 'try' keyword
        text.count('except'),                              # 'except' keyword
        len(re.findall(r'\d+\.\d+', text))                 # float literals
    ]
    return np.array(features, dtype=float)


#Predict function for sklearn
def predict_ai_generated(code_snippet):
    # Embedding
    embedding = model.encode([code_snippet])[0]  
    # Stylometric features
    style = extract_stylometric_features(code_snippet)  
    # Combine
    features = np.hstack((embedding, style)).reshape(1, -1)  

    # Predict
    prob = clf.predict_proba(features)[0][1]
    return prob


@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        print(f"Received request: {data}")

        code = data.get('code', '')
        question = data.get('question', {})
        tags = question.get('tags', [])
        print(f"Received tags: {tags}")

        if not code:
            return jsonify({"error": "No code provided"}), 400

        if not any(tag in ALLOWED_TAGS for tag in tags):
            print(f"Ignored: Tags {tags} not in allowed set {ALLOWED_TAGS}")
            return jsonify({"error": "Tag not allowed"}), 403

        load_model()
        probability = predict_ai_generated(code)

        result = {
            "ai_probability": float(probability),     
            "is_ai_generated": bool(probability > 0.5) 
        }
        print(f"Sending response: {result}")
        return jsonify(result)

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Start server
if __name__ == '__main__':
    print("Starting Flask backend for DetectAI...")
    load_model()
    app.run(host='127.0.0.1', port=5000, debug=True)
