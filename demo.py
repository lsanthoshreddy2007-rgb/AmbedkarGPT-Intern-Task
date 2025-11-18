from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import argparse
import os


def build_vectorstore(text_path: str, persist_dir: str = './chroma_demo', chunk_size: int = 500, overlap: int = 50):
    # Load the text file manually to avoid version differences in langchain loaders
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    docs = [Document(page_content=text, metadata={'source': text_path})]
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_documents(docs)
    embed = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectordb = Chroma.from_documents(chunks, embed, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb


def run_demo(args):
    text_path = args.speech
    db_dir = args.db
    if not os.path.exists(db_dir) or args.rebuild:
        print('Building demo vectorstore...')
        vectordb = build_vectorstore(text_path, persist_dir=db_dir, chunk_size=args.chunk_size)
    else:
        print('Loading existing demo vectorstore...')
        embed = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vectordb = Chroma(persist_directory=db_dir, embedding_function=embed)

    queries = args.queries or ["What is the main argument?"]
    for q in queries:
        print('\n=== QUERY:', q)
        docs = vectordb.similarity_search(q, k=args.k)
        print(f'Retrieved {len(docs)} chunks:')
        for i, d in enumerate(docs, 1):
            text = getattr(d, 'page_content', str(d))
            src = getattr(d, 'metadata', {}).get('source', 'unknown')
            print(f'-- Chunk {i} (source: {src}) --')
            print(text.strip())
            print('----------------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--speech', default='speech.txt')
    parser.add_argument('--db', default='./chroma_demo')
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--chunk_size', type=int, default=500)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--queries', nargs='*')
    args = parser.parse_args()
    run_demo(args)
