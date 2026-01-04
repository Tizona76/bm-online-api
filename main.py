from fastapi import FastAPI

app = FastAPI(title="BM Online API", version="0.0.1")

@app.get("/v1/health")
@app.get("/")
def root():
    return {"ok": True, "service": "bm-online-api"}
def health():
    return {"ok": True}
