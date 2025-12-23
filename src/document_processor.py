from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

class DocumentProcessor:
    def __init__(self, drive_path):
        self.drive_path = drive_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
    
    def process_documents(self, pdf_files):
        all_docs = []
        for pdf in pdf_files:
            pdf_path = os.path.join(self.drive_path, pdf)
            loader = PyPDFLoader(pdf_path)
            documents = loader.load_and_split(self.text_splitter)
            all_docs.extend(documents)
        
        vectorstore = Chroma.from_documents(
            documents=all_docs,
            embedding=self.embeddings,
            persist_directory=os.path.join(self.drive_path, "vectorstore")
        )
        vectorstore.persist()
        return vectorstore
