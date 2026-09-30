import os
os.environ.setdefault("CODEDOCTOR_TIER", "interactive")
import uvicorn

if __name__ == "__main__":
    print("==================================================")
    print("Starting CodeDoctor Server")
    print("Frontend & Workbench: http://127.0.0.1:8077")
    print("Backend API & Docs:   http://127.0.0.1:8077/docs")
    print("==================================================")
    uvicorn.run("app.server:app", host="127.0.0.1", port=8077, reload=False)
