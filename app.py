from flask import Flask, request, jsonify
from flask_cors import CORS
import os, json, glob, re
import requests
import numpy as np

app = Flask(__name__)
CORS(app)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_EMBED_URL = "https://api.openai.com/v1/embeddings"

# =========================
#  RAG: Indexação de documentos
# =========================

DOCS_DIR = os.environ.get("DOCS_DIR", "docs")
CHUNK_CHARS = 1800
CHUNK_OVERLAP = 200
TOP_K = 12                # mais trechos
SCORE_THRESHOLD = 0.40    # mais tolerante
STRICT_MODE = True

index = []

def clean_text(t: str) -> str:
    t = t.replace("\r", "")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def load_docs() -> list[dict]:
    docs = []
    for path in glob.glob(os.path.join(DOCS_DIR, "*")):
        if path.lower().endswith((".txt", ".md")) and not path.lower().startswith("readme"):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = clean_text(f.read())
                docs.append({"source": os.path.basename(path), "text": text})
    return docs

def chunk_text(text: str, src: str) -> list[dict]:
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i:i+CHUNK_CHARS]
        last_nl = chunk.rfind("\n")
        if last_nl > CHUNK_CHARS * 0.5:
            chunk = chunk[:last_nl]
        chunks.append({"source": src, "text": chunk.strip()})
        i += max(len(chunk) - CHUNK_OVERLAP, 1)
    return chunks

def embed_texts(texts: list[str]) -> np.ndarray:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": OPENAI_EMBEDDING_MODEL,
        "input": texts
    }
    r = requests.post(OPENAI_EMBED_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    vecs = [np.array(d["embedding"], dtype=np.float32) for d in data["data"]]
    vecs = [v / (np.linalg.norm(v) + 1e-10) for v in vecs]
    return np.vstack(vecs)

def build_index():
    global index
    docs = load_docs()
    parts = []
    for d in docs:
        parts.extend(chunk_text(d["text"], d["source"]))
    if not parts:
        index = []
        return
    embeddings = embed_texts([p["text"] for p in parts])
    index = [{"text": p["text"], "source": p["source"], "embedding": e} for p, e in zip(parts, embeddings)]

def similarity_search(query: str, top_k: int = TOP_K):
    if not index:
        return []
    q_vec = embed_texts([query])[0]
    sims = []
    for i, item in enumerate(index):
        sim = float(np.dot(q_vec, item["embedding"]))
        sims.append((sim, i))
    sims.sort(reverse=True, key=lambda x: x[0])
    results = []
    for sim, i in sims[:top_k]:
        results.append({"score": sim, "text": index[i]["text"], "source": index[i]["source"]})
    return results

def build_context(query: str):
    hits = similarity_search(query, TOP_K)
    if not hits:
        return []
    if STRICT_MODE:
        filtered = [h for h in hits if h["score"] >= SCORE_THRESHOLD]
        if filtered:
            return filtered
        # fallback: se nada passou, pega os 3 melhores
        return hits[:3]
    return hits

SYSTEM_PROMPT = (
    "Você é um assistente do Programa Ciência e Gestão pela Educação (PCGE). "
    "Responda SOMENTE com base nos trechos de CONTEXTO fornecidos. "
    "Se a pergunta não tiver suporte suficiente, diga educadamente: "
    "\"No momento, essa informação não está disponível nos documentos do PCGE.\"\n\n"
    "Regras:\n"
    "- Não invente dados; não use conhecimento externo.\n"
    "- Responda em português do Brasil, de forma clara e objetiva.\n"
)

def make_user_prompt(user_msg: str, context_slices: list[dict]) -> str:
    if not context_slices:
        contexto = "(nenhum trecho relevante encontrado)"
    else:
        parts = []
        for i, h in enumerate(context_slices, start=1):
            parts.append(f"[Trecho {i} | fonte: {h['source']} | score: {h['score']:.2f}]\n{h['text']}\n")
        contexto = "\n".join(parts)

    prompt = (
        f"PERGUNTA DO USUÁRIO:\n{user_msg}\n\n"
        f"CONTEXTO (trechos dos documentos):\n{contexto}\n\n"
        "INSTRUÇÕES AO ASSISTENTE:\n"
        "- Responda somente com base no CONTEXTO acima.\n"
        "- Se não houver base suficiente, diga: "
        "\"No momento, essa informação não está disponível nos documentos do PCGE.\""
    )
    return prompt

def call_openai_chat(messages):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 700
    }
    r = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_msg = (data.get("message") or "").strip()

    if not user_msg:
        return jsonify({"reply": "Por favor, envie uma pergunta."})

    context_slices = build_context(user_msg)
    if STRICT_MODE and not context_slices:
        return jsonify({"reply": "No momento, essa informação não está disponível nos documentos do PCGE."})

    user_prompt = make_user_prompt(user_msg, context_slices)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    try:
        reply = call_openai_chat(messages)
        return jsonify({"reply": reply})
    except Exception as e:
        print("Erro OpenAI:", str(e))
        return jsonify({"reply": "Não consegui responder agora. Tente novamente em instantes."}), 200

@app.route("/admin/reindex", methods=["POST", "GET"])
def reindex():
    try:
        build_index()
        return jsonify({"status": "ok", "chunks": len(index)})
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500

@app.route("/debug/search", methods=["GET"])
def debug_search():
    q = request.args.get("q", "")
    hits = similarity_search(q, TOP_K)
    return jsonify(hits)

@app.route("/")
def health():
    return "OK", 200

build_index()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))










