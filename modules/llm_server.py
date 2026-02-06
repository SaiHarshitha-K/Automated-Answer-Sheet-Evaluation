# llm_server.py
# Offline Phi-3 Mini LLM Server (Stable)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
import time

# ============================================================
# HARD-CODED MODEL PATH (NO ENV VARS)
# ============================================================
MODEL_PATH = r"D:\llm_check\Phi-3-mini-4k-instruct-q4.gguf"

# ============================================================
# LLAMA SETTINGS (CPU SAFE)
# ============================================================
N_CTX = 4096
N_THREADS = 8
N_BATCH = 256

app = FastAPI(title="Offline Phi-3 LLM Server")

llm = None


class ChatReq(BaseModel):
    system: str
    user: str
    max_tokens: int = 400
    temperature: float = 0.0
    top_p: float = 1.0
    repeat_penalty: float = 1.05


@app.on_event("startup")
def load_model():
    global llm
    print("🔧 Loading Phi-3 Mini model...")
    t0 = time.time()

    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_batch=N_BATCH,
        verbose=False,
    )

    print(f"✅ Model loaded in {time.time() - t0:.1f}s")


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": MODEL_PATH,
        "ctx": N_CTX,
    }


@app.post("/chat")
def chat(req: ChatReq):
    if llm is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": req.system},
                {"role": "user", "content": req.user},
            ],
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            repeat_penalty=req.repeat_penalty,
        )
        return {"content": resp["choices"][0]["message"]["content"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
