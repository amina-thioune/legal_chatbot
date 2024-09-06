from langchain_text_splitters import NLTKTextSplitter
import nltk


def nltk_text_splitter(text, chunk_size, chunk_overlap) : 
    nltk.download('punkt')
    text_splitter = NLTKTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    texts = text_splitter.split_documents(text)
    return texts