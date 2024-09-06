from langchain_text_splitters import SpacyTextSplitter
import spacy


def spacy_text_splitter(text, chunk_size, chunk_overlap) : 
    spacy.load('en_core_web_sm')
    text_splitter = SpacyTextSplitter(chunk_size= chunk_size, chunk_overlap = chunk_overlap)
    texts = text_splitter.split_documents(text)
    return texts 