import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
app = Flask(__name__)
CORS(app)  # permite que o front acesse de outro domínio

@app.get("/")
def health():
    return "OK", 200

@app.post("/api/chat")
def chat():
    data = request.get_json(force=True)
    user_msg = (data or {}).get("message", "").strip()
    if not user_msg:
        return jsonify({"error": "Mensagem vazia"}), 400

    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "Você é o assistente 'ChatGPT - PCGE'."},
                {"role": "user", "content": user_msg}
            ],
        },
        timeout=60
    )
    r.raise_for_status()
    reply = r.json()["choices"][0]["message"]["content"]
    return jsonify({"reply": reply})
