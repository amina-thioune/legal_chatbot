import os
os.environ["AI21_API_KEY"] = "b3hX5sdHCBhEEn3oah2t2q9v6wmnC38t"
from langchain_ai21 import AI21SemanticTextSplitter

def AI21_text_splitter(text):
    semantic_text_splitter = AI21SemanticTextSplitter()
    texts = semantic_text_splitter.create_documents(text)
    return texts
