from langchain_text_splitters import RecursiveCharacterTextSplitter

def recursive_character_text_splitter(text, chunk_size, chunk_overlap) : 
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
    is_separator_regex=False,
    )
    texts = text_splitter.create_documents(text)
    return texts