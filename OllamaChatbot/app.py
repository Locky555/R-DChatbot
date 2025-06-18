from flask import Flask, request, jsonify, render_template
from multi_rag import MultiRAG
import requests
import re

app = Flask(__name__)
rag = MultiRAG(docs_folder="docs")

OLLAMA_API_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "deepseek-r1:8b"

def clean_ollama_response(text):
    # Remove any "<thinking>" or similar tags
    return re.sub(r"<[^>]+>", "", text).strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    print("Received from frontend:", user_input)

    context = rag.retrieve_relevant_context(user_input)

    prompt = f"""You are a helpful assistant. Answer the question using the provided context.

Context:
{context}

Question: {user_input}
Answer:"""

    try:
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }
        )

        full_response = response.json()
        raw_answer = full_response.get('message', {}).get('content', 'No content in response')
        answer = clean_ollama_response(raw_answer)

        return jsonify({"response": answer})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"response": "An error occurred connecting to Ollama."})

if __name__ == '__main__':
    app.run(debug=True)
