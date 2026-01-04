from fastapi import FastAPI

app = FastAPI(title="BM Online API", version="0.0.1")

@app.get("/v1/health")
def health():
    return {"ok": True}
