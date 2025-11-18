
"""evaluation.py
Evaluation framework for Assignment-2.
This script loads the corpus and test_dataset.json, runs retrieval+generation for each question,
and computes retrieval and answer-quality metrics for three chunking strategies.
It's a template that uses LangChain, ChromaDB, HuggingFaceEmbeddings and Ollama.
"""
import json
import os
import math
from collections import defaultdict
from pprint import pprint

from langchain.document_loaders import TextLoader
from langchain.text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

# Optional metrics libraries
try:
    from rouge_score import rouge_scorer
except Exception:
    rouge_scorer = None
try:
    from nltk.translate.bleu_score import sentence_bleu
except Exception:
    sentence_bleu = None
try:
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    cosine_similarity = None

DATA_DIR = './corpus'
TEST_FILE = './test_dataset.json'
RESULTS_OUT = './test_results.json'

CHUNK_CONFIGS = {
    'small': {'chunk_size': 250, 'overlap': 50},
    'medium': {'chunk_size': 550, 'overlap': 100},
    'large': {'chunk_size': 900, 'overlap': 150},
}

def load_test_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)['test_questions']

def build_vectordb_from_corpus(chunk_size, overlap, persist_dir):
    # Concatenate all corpus files into documents
    docs = []
    for fn in sorted(os.listdir(DATA_DIR)):
        if fn.lower().endswith('.txt'):
            loader = TextLoader(os.path.join(DATA_DIR, fn), encoding='utf-8')
            docs.extend(loader.load())
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_documents(docs)
    embed = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectordb = Chroma.from_documents(chunks, embed, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb

def make_qa_chain(vectordb):
    llm = Ollama(model='mistral')
    retriever = vectordb.as_retriever(search_type='similarity', search_kwargs={'k':5})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever, return_source_documents=True)
    return qa

def compute_basic_retrieval_metrics(retrieved_docs, ground_truth_docs):
    # retrieved_docs: list of filenames returned by retriever in ranked order
    # ground_truth_docs: list of filenames (expected)
    hit = any(d in ground_truth_docs for d in retrieved_docs)
    # MRR: rank of first correct doc
    rr = 0.0
    for i, d in enumerate(retrieved_docs):
        if d in ground_truth_docs:
            rr = 1.0 / (i + 1)
            break
    return {'hit': int(hit), 'reciprocal_rank': rr}

def evaluate_chunking():
    tests = load_test_dataset(TEST_FILE)
    overall_results = {}
    for cfg_name, cfg in CHUNK_CONFIGS.items():
        print('Running chunk config:', cfg_name)
        persist_dir = f'./chroma_{cfg_name}'
        if os.path.exists(persist_dir):
            # reuse existing DB
            vectordb = Chroma(persist_directory=persist_dir, embedding_function=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'))
        else:
            vectordb = build_vectordb_from_corpus(cfg['chunk_size'], cfg['overlap'], persist_dir)
        qa = make_qa_chain(vectordb)
        results = []
        for q in tests:
            q_text = q['question']
            ground = q['ground_truth']
            resp = qa(q_text)
            answer_text = resp.get('result', '').strip()
            # retrieved docs metadata -> try to extract file names
            retrieved_files = []
            for d in resp.get('source_documents', []):
                src = getattr(d, 'metadata', {}).get('source')
                if src:
                    retrieved_files.append(os.path.basename(src))
            rmetrics = compute_basic_retrieval_metrics(retrieved_files, q.get('source_documents', []))
            # placeholder for ROUGE, BLEU, cosine etc.
            rouge_L = None
            bleu = None
            cos_sim = None
            if rouge_scorer:
                scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
                rouge_L = scorer.score(ground, answer_text)['rougeL'].fmeasure
            if sentence_bleu:
                try:
                    refs = [ground.split()]
                    bleu = sentence_bleu(refs, answer_text.split())
                except Exception:
                    bleu = 0.0
            # append result
            results.append({
                'id': q['id'],
                'question': q_text,
                'ground_truth': ground,
                'answer': answer_text,
                'retrieved_files': retrieved_files,
                'hit': rmetrics['hit'],
                'reciprocal_rank': rmetrics['reciprocal_rank'],
                'rougeL': rouge_L,
                'bleu': bleu,
            })
        overall_results[cfg_name] = results
        # Save intermediate results
        with open(f'./test_results_{cfg_name}.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
    # Save summary
    with open(RESULTS_OUT, 'w', encoding='utf-8') as f:
        json.dump(overall_results, f, indent=2)
    print('Evaluation complete. Results saved to', RESULTS_OUT)

if __name__ == '__main__':
    evaluate_chunking()
