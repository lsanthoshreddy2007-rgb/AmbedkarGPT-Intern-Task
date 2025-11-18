from sentence_transformers import SentenceTransformer
import numpy as np
import argparse


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        if end == L:
            break
        start = max(0, end - overlap)
    return chunks


def embed_texts(model, texts, batch_size=32):
    embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings


def cosine_sim(a, b):
    return np.dot(a, b)


def run_local_demo(args):
    with open(args.speech, 'r', encoding='utf-8') as f:
        text = f.read()
    chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f'Created {len(chunks)} chunks.')
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    chunk_embeds = embed_texts(model, chunks)

    queries = args.queries or ["What is the problem of caste?"]
    for q in queries:
        q_emb = embed_texts(model, [q])[0]
        sims = np.dot(chunk_embeds, q_emb)
        idx = np.argsort(-sims)[:args.k]
        print('\n=== QUERY:', q)
        for rank, i in enumerate(idx, 1):
            print(f'-- Rank {rank}, score={sims[i]:.4f} --')
            print(chunks[i].strip())
            print('----------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--speech', default='speech.txt')
    parser.add_argument('--chunk_size', type=int, default=500)
    parser.add_argument('--overlap', type=int, default=50)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--queries', nargs='*')
    args = parser.parse_args()
    run_local_demo(args)
