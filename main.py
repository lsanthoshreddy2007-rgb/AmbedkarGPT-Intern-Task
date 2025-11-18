
"""main.py - AmbedkarGPT Intern Task (Assignment-1)
Simple RAG prototype using LangChain, ChromaDB, HuggingFaceEmbeddings and Ollama.
NOTE: This script is a template and assumes you have installed Ollama, ChromaDB and the Python packages.
Configure paths and parameters as needed.
"""
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
import argparse
import os

def build_vectorstore(text_path: str, persist_dir: str = './chroma_db', chunk_size: int = 500, overlap: int = 50):
    loader = TextLoader(text_path, encoding='utf-8')
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_documents(docs)
    embed = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectordb = Chroma.from_documents(chunks, embed, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb

def make_qa_chain(vectordb):
    # Ollama LLM integration; ensure ollama and Mistral are installed locally
    llm = Ollama(model='mistral')
    retriever = vectordb.as_retriever(search_type='similarity', search_kwargs={'k':3})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever, return_source_documents=True)
    return qa

def run_cli(args):
    text_path = args.speech if args.speech else 'speech.txt'
    db_dir = args.db if args.db else './chroma_db'
    if not os.path.exists(db_dir) or args.rebuild:
        print('Building vectorstore...')
        vectordb = build_vectorstore(text_path, persist_dir=db_dir, chunk_size=args.chunk_size)
    else:
        print('Loading existing vectorstore...')
        embed = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vectordb = Chroma(persist_directory=db_dir, embedding_function=embed)
    qa = make_qa_chain(vectordb)
    print('Ready. Ask questions (type "exit" to quit).')
    while True:
        q = input('> ')
        if q.strip().lower() in ('exit', 'quit'):
            break
        resp = qa(q)
        print('\n--- ANSWER ---')
        print(resp['result'])
        print('\n--- SOURCES ---')
        for d in resp.get('source_documents', []):
            print('-', getattr(d, 'metadata', {}).get('source', 'unknown'))
        print('---------------\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--speech', help='Path to speech.txt', default='speech.txt')
    parser.add_argument('--db', help='Chroma DB persist directory', default='./chroma_db')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild the vectorstore')
    parser.add_argument('--chunk_size', type=int, default=500, help='Character chunk size')
    args = parser.parse_args()
    run_cli(args)
