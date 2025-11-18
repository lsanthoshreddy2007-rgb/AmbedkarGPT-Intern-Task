
# AmbedkarGPT-Intern-Task

This repository contains two assignment deliverables for the Kalpit Pvt Ltd AI intern task:
- Assignment-1: a simple RAG prototype (main.py)
- Assignment-2: an evaluation framework (evaluation.py)

## Repository structure
```
AmbedkarGPT-Intern-Task/
  ├─ main.py
  ├─ evaluation.py
  ├─ requirements.txt
  ├─ README.md
  ├─ speech.txt
  ├─ corpus/
  │    ├─ speech1.txt
  │    └─ ...
  ├─ test_dataset.json
  ├─ test_results.json
```

## Quick start

1. Create a Python 3.8+ virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Install Ollama and pull the Mistral model:
   Follow instructions at https://ollama.ai (install locally) and run:
   ```
   ollama pull mistral
   ```

3. Run the RAG CLI:
   ```
   python main.py --speech speech.txt --rebuild
   ```

4. Run the evaluation (this will build vectorstores per chunk config and run tests):
   ```
   python evaluation.py
   ```

## Notes & Troubleshooting
- Ollama must be installed locally and accessible from your PATH.
- ChromaDB path is persisted under `./chroma_*` directories.
- Some metrics (ROUGE, BLEU) require installing additional packages (see requirements.txt).
- The evaluation scripts are templates: adjust `CHUNK_CONFIGS` and retrieval `k` as needed.

## What I included
- Fully populated `corpus/` with 6 small speech files (from test dataset)
- `test_dataset.json` following the required format
- Templates for both `main.py` and `evaluation.py`

If you want, I can:
- Push these files directly to your GitHub (you will need to provide a repo URL and a token), OR
- Generate a zip of the repository so you can upload manually.
