import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class MultiRAG:
    def __init__(self, docs_folder="docs", db_path="faiss_index_multi"):
        self.docs_folder = docs_folder
        self.db_path = db_path
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None
        self.load_and_embed_pdfs()

    def load_and_embed_pdfs(self):
        print("Rebuilding FAISS index...")  # Always rebuild for now
        documents = []

        # DEBUG: Print every PDF file being loaded
        for filename in os.listdir(self.docs_folder):
            if filename.endswith(".pdf"):
                print(f"Loading PDF: {filename}")  # <--- Debug print
                loader = PyPDFLoader(os.path.join(self.docs_folder, filename))
                documents.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.vectorstore.save_local(self.db_path)
        print("Built and saved new FAISS index with multiple PDFs.")

    def retrieve_relevant_context(self, query, k=30):
        docs = self.vectorstore.similarity_search(query, k=k)
        return "\n".join([doc.page_content for doc in docs])
