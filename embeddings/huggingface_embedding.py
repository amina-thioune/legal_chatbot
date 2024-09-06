
from langchain_community.embeddings import HuggingFaceEmbeddings


def huggingface_embedding(model_name, docs_after_split) : 
    huggingface_embeddings = HuggingFaceEmbeddings(
        model_name= model_name, 
        model_kwargs={'device':'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
  
    return huggingface_embeddings


