"""Run RAG: build indexes (if needed) then answer a test query. Install deps first: pip install -r requirements.txt"""
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
# Load .env first so OPENAI_API_KEY is set before any other code runs
load_dotenv(dotenv_path=str(ROOT / ".env"), encoding="utf-8")
sys.path.insert(0, str(ROOT))

def main():
    # Query: from command line, or type when prompted
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:]).strip()
    else:
        query = input("Enter your question: ").strip()
    if not query:
        query = "What is the refund policy?"
        print("(Using default question:", query, ")")
    # 1. Build indexes
    print("Building indexes...")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "phase0_build_indexes.py")], cwd=str(ROOT), check=False)
    # 2. Run pipeline
    import os
    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        print("OPENAI_API_KEY not found. Check that .env exists in", ROOT, "with line: OPENAI_API_KEY=sk-...")
        return 1
    print("Running RAG pipeline...")
    from src.rag import RAGPipeline
    try:
        from config.settings import LOG_DIR
        log_dir = LOG_DIR
    except Exception:
        log_dir = ROOT / "data" / "processed" / "logs"
    p = RAGPipeline(log_dir=log_dir)
    out = p.run(query)
    print("Query:", out["query"])
    print("Chunks retrieved:", len(out["chunks"]))
    print("Answer:", out["answer"] or "(no answer)")
    # Collect feedback so we can train the router better later (phase2/phase3)
    decision = out.get("decision")
    if decision and p.log_dir:
        feedback = input("Was this answer helpful? (y/n, or Enter to skip): ").strip().lower()
        if feedback in ("y", "yes"):
            p.record_feedback(query, decision, 1.0)
            print("Thanks! Feedback recorded (reward=1).")
        elif feedback in ("n", "no"):
            p.record_feedback(query, decision, 0.0)
            print("Feedback recorded (reward=0). Run phase3 bandit replay or phase2 retrain to improve.")
        else:
            print("(No feedback recorded)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
