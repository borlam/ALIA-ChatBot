from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

class ChatEngine:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.qa_chain = None
        
    def initialize_model(self, api_key):
        llm = HuggingFaceHub(
            repo_id="jarvisx17/Jais-13B-Chat",
            huggingfacehub_api_token=api_key,
            model_kwargs={
                "temperature": 0.1,
                "max_new_tokens": 512,
                "do_sample": True
            }
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            return_source_documents=True
        )
    
    def query(self, question):
        if not self.qa_chain:
            return "Modelo no inicializado"
        result = self.qa_chain({"query": question})
        return result['result'], result['source_documents']
