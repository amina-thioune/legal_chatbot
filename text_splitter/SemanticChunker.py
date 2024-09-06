from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceBgeEmbeddings



def semantic_chunker(text, breakpoint_threshold_type) : 

    huggingface_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",  
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings': True}
    )
    text_splitter = SemanticChunker(
    huggingface_embeddings,
    breakpoint_threshold_type=breakpoint_threshold_type
    )
    texts = text_splitter.create_documents(text)
    return texts