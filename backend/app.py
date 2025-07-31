from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import warnings
import numpy as np
import re
from threading import Lock
from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# === Globals ===
tokenizer = None
codebert_model = None
clf = None
model_lock = Lock()
MODEL_PATH = "CPU_human_ai_classifier.pkl"

ALLOWED_TAGS = {"python", "java"}  

JAVA_PYTHON_KEYWORDS = set([
    "def", "return", "if", "else", "elif", "while", "for", "break", "continue", "try", "except",
    "import", "from", "as", "class", "pass", "with", "yield", "lambda", "global", "nonlocal", "assert",
    "public", "private", "protected", "static", "final", "void", "int", "double", "float", "char", "boolean",
    "new", "catch", "finally", "throws", "throw", "switch", "case", "package", "interface", "implements", "extends"
])

#Stylometric features extraction
def extract_stylometric_features(code: str) -> np.ndarray:
    if not isinstance(code, str) or not code.strip():
        return np.zeros(17, dtype=np.float32)

    lines = code.split('\n')
    num_lines = len(lines)
    line_lengths = [len(line) for line in lines]
    avg_line_length = np.mean(line_lengths) if lines else 0
    blank_lines = sum(1 for line in lines if not line.strip())

    tokens = re.findall(r'\b\w+\b', code)
    num_tokens = len(tokens)
    avg_token_length = np.mean([len(tok) for tok in tokens]) if tokens else 0
    num_keywords = sum(1 for tok in tokens if tok in JAVA_PYTHON_KEYWORDS)
    keyword_ratio = num_keywords / num_tokens if num_tokens else 0

    comment_lines = sum(1 for line in lines if re.match(r'^\s*(#|//|/\*|\*)', line.strip()))
    comment_ratio = comment_lines / num_lines if num_lines else 0

    num_assignments = len(re.findall(r'\w+\s*=+', code))
    num_function_defs = len(re.findall(r'\b(def|void|public\s+|private\s+|protected\s+).*?\(', code))

    whitespace_ratio = len(re.findall(r'\s', code)) / len(code) if code else 0
    uses_tabs = int('\t' in code)

    indent_levels = [len(re.match(r'^\s*', line).group()) for line in lines if line.strip()]
    indent_variance = np.var(indent_levels) if indent_levels else 0
    max_indent_level = max(indent_levels) if indent_levels else 0

    num_brackets = sum(code.count(b) for b in '{}()[]')

    return np.array([
        num_lines,
        avg_line_length,
        blank_lines,
        num_tokens,
        avg_token_length,
        num_keywords,
        keyword_ratio,
        comment_lines,
        comment_ratio,
        num_assignments,
        num_function_defs,
        whitespace_ratio,
        uses_tabs,
        indent_variance,
        max_indent_level,
        num_brackets,
        len(code)
    ], dtype=np.float32)

def load_model():
    global tokenizer, codebert_model, clf
    with model_lock:
        if tokenizer is None or codebert_model is None or clf is None:
            print("Loading CodeBERT and classifier...")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            codebert_model = AutoModel.from_pretrained("microsoft/codebert-base")
            with open(MODEL_PATH, "rb") as f:
                clf = pickle.load(f)
            print("All models loaded!")

def get_codebert_embedding(code):
    inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = codebert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # CLS token

def predict_ai_generated(code_snippet):
    embedding = get_codebert_embedding(code_snippet)
    style_features = extract_stylometric_features(code_snippet)
    features = np.hstack((embedding, style_features)).reshape(1, -1)
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


if __name__ == '__main__':
    print("Starting Flask backend for DetectAI...")
    load_model()
    app.run(host='127.0.0.1', port=5000, debug=True)
