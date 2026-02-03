import os
import faiss
import numpy as np
import multiprocessing
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

PDF_PATH = "docs.pdf"
INDEX_PATH = "faiss_index"
OPENROUTER_API_KEY = "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"


 # Embedding Wrapper
class MiniLMEmbeddings(Embeddings):
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts,
            batch_size=256,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu"
                                                                             "")
embeddings = MiniLMEmbeddings(model)


    # FAISS Helpers
def set_nprobe(index, nprobe: int = 10):
    if hasattr(index, 'sub_index'):
        set_nprobe(index.sub_index, nprobe)
    elif hasattr(index, 'nprobe'):
        index.nprobe = nprobe
        print(f"nprobe set to {nprobe}")
    else:
        print("Warning: could not find an IVF index to set nprobe on.")


    # Load and split into chunks
def load_and_chunk(pdf_path: str):
    loader = PyMuPDFLoader(pdf_path)
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"Chunks created: {len(chunks)}")
    return chunks


    # Create Embeddings

def create_embeddings(chunks: list):
    texts = [chunk.page_content for chunk in chunks]
    embeddings_array = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")
    return embeddings_array


    # Build FAISS Index

def build_faiss_index(embeddings_array: np.ndarray):
    faiss.omp_set_num_threads(multiprocessing.cpu_count())

    dimension = embeddings_array.shape[1]       # 384
    n_vectors = len(embeddings_array)
    nlist = max(1, int(np.sqrt(n_vectors)))

    quantizer = faiss.IndexFlatIP(dimension)
    base_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

    # Wrap in IDMap — this is what makes reconstruct() work after save/load
    index = faiss.IndexIDMap(base_index)

    # Train the base IVF index
    base_index.train(embeddings_array)

    # Add with explicit IDs (0, 1, 2, ...)
    ids = np.arange(n_vectors, dtype=np.int64)
    index.add_with_ids(embeddings_array, ids)

    # Set nprobe on the underlying IVF index
    base_index.nprobe = 10

    print(f"FAISS index built: {index.ntotal} vectors, {nlist} clusters")
    return index


    # Build vectorstore

def build_vectorstore(index, chunks):
    docstore_dict = {str(i): chunks[i] for i in range(len(chunks))}
    index_to_docstore_id = {i: str(i) for i in range(len(chunks))}

    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(docstore_dict),
        index_to_docstore_id=index_to_docstore_id
    )
    return vectorstore

    #Save and load faiss index

def get_vectorstore(pdf_path: str, index_path: str):
    if os.path.exists(index_path):
        print(f"Loading existing index from '{index_path}'...")
        vectorstore = FAISS.load_local(
            index_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        # Re-set nprobe on the underlying IVF index after load
        set_nprobe(vectorstore.index)
        print(f"Index loaded: {vectorstore.index.ntotal} vectors")
    else:
        print("No saved index found. Building from scratch...")
        chunks = load_and_chunk(pdf_path)
        embeddings_array = create_embeddings(chunks)
        index = build_faiss_index(embeddings_array)
        vectorstore = build_vectorstore(index, chunks)

        vectorstore.save_local(index_path)
        print(f"Index saved to '{index_path}'")

    return vectorstore

    # RAG Chain

def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using  the context below. Use your additional knowledge if necessary\n\nContext: {context}"),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(
        model="nvidia/nemotron-3-nano-30b-a3b:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY
    )

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# Main
def main():
    vectorstore = get_vectorstore(PDF_PATH, INDEX_PATH)
    rag_chain = build_rag_chain(vectorstore)

    print("\n RAG ready. Type your question (or 'quit' to exit) \n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        print("Assistant: ", end="", flush=True)
        for chunk in rag_chain.stream(question):
            print(chunk, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    main()