import streamlit as st
from io import StringIO
from models.mistral import load_mistral
from models.saulm import load_model
from models.summarization import llm_oriented_summarization
from text_splitter.RecursiveCharacterTextSplitter import  recursive_character_text_splitter
from text_splitter.CharacterTextSplitter import  character_text_splitter
from text_splitter.SemanticChunker import semantic_chunker
from text_splitter.NLTKTextSplitter import nltk_text_splitter
from text_splitter.SpacyTextSplitter import spacy_text_splitter
from text_splitter.AI21SemanticTextSplitter import AI21_text_splitter
from embeddings.huggingface_embedding import huggingface_embedding
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA 
from langchain_core.documents import Document


# Configure the page to use the entire width of the browser window
st.set_page_config(layout="wide")

# Set the title of the app
st.title("💬 Chatbot")

# Sidebar for selecting model categories
model_category = st.sidebar.selectbox(
    "Model Categories",
    ["Multilingual Model", "Model Oriented Summarization"]
)

# Display models based on the selected category
if model_category == "Model Oriented Summarization":
    model_oriented_summarization = st.sidebar.radio(
        "Models:",
        ["Roberta_BART_fixed_V1", "LexLM_Longformer_BART_fixed_V1", "T5_no_extraction_V1", "RoBERTa_BART_dependent_V1"],
        index=None
    )


# Display models based on the selected category
elif model_category == "Multilingual Model":
    model = st.sidebar.radio("Models:",["Mistral", "SaulM"],index=None)

    # Show a quantization method dropdown if the model is "SaulM".
    if model =="SaulM":
        method = st.sidebar.selectbox("Quantization :", ["4 bits","8 bits","All"], index=None)


    # Sidebar for selecting the option between RAG and Base
    option = st.sidebar.radio("Choose option", ["RAG", "Base"], index=None)

    # If the option selected is "RAG"
    if option == "RAG":

        # Upload a multiple files
        uploader_files = st.sidebar.file_uploader("Choose file", accept_multiple_files=True)

        # To convert to a string based IO:
        files = []
        sources = []
        for uploaded_file in uploader_files : 
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            files.append(stringio)
            sources.append(uploaded_file.name)

        docs_before_split_document = []
        docs_before_split = []
        # To read file as string:
        if files is not None : 
            for file, source in zip(files, sources) :
                string_data = file.read()
                docs_before_split_document.append(Document(page_content=string_data, metadata={"source":source}))
                docs_before_split.append(string_data)

        # Additional sidebar options for "RAG"
        text_splitter = st.sidebar.selectbox("Text Splitter:",
                ["CharacterTextSplitter","RecursiveCharacterTextSplitter", "SemanticChunker", "AI21SemanticTextSplitter", "NLTKTextSplitter", "SpacyTextSplitter" ],
                index=None)
        if text_splitter =="CharacterTextSplitter" or text_splitter == "RecursiveCharacterTextSplitter" or text_splitter == "NLTKTextSplitter"  or text_splitter == "SpacyTextSplitter": 
            chunk_size = st.sidebar.slider("Chunk size", 50, 1000, 50)
            chunk_overlap = st.sidebar.slider("Chunk overlap", 0, 100, 0)
        elif text_splitter == "SemanticChunker" : 
            breakpoint_threshold_type = st.sidebar.selectbox("Breakpoint_thresold_type:", ["percentile", "standard_deviation", "interquartile", "gradient"], index=None)

        # Multilingual Embeddings
        multilingual_embeddings = st.sidebar.radio("Multilingual Embeddings:",
                ["bge-m3", "multilingual-e5-large-instruct"],
                index=None)
        
        # French Embeddings
        french_embeddings = st.sidebar.radio("or French Embeddings:",
                ["Sentence-CamemBERT-Large", "FlauBERT", "BartHez"],
                index=None)
        kwargs = st.sidebar.slider("Top K : ", 1, 10, 1 )

# Run button in the sidebar
run = st.sidebar.button("Run", type="primary", use_container_width=True)


if run : 
    
    # if we choose a multilingual model
    if model_category == "Multilingual Model" : 
        # load Mistral 
        if model == "Mistral" :
            llm = load_mistral()
            st.session_state.llm = llm

        # load SaulM
        elif model =="SaulM" : 
            llm, tokenizer, device = load_model(method=method)
            st.session_state.llm = llm
   

        # if we choose RAG
        if option == "RAG" : 

            # Choose text splitter
            if text_splitter == "CharacterTextSplitter" : 
                docs_after_split = character_text_splitter(docs_before_split, chunk_size, chunk_overlap)          
            elif text_splitter == "RecursiveCharacterTextSplitter" : 
                docs_after_split = recursive_character_text_splitter(docs_before_split, chunk_size, chunk_overlap)
            elif text_splitter == "SemanticChunker" :
                docs_after_split = semantic_chunker(docs_before_split, breakpoint_threshold_type)
            elif text_splitter == "AI21SemanticTextSplitter" : 
                docs_after_split = AI21_text_splitter(docs_before_split)
            elif text_splitter == "NLTKTextSplitter" : 
                docs_after_split = nltk_text_splitter(docs_before_split_document, chunk_size, chunk_overlap)  
            elif text_splitter == "SpacyTextSplitter" :
                docs_after_split = spacy_text_splitter(docs_before_split_document, chunk_size, chunk_overlap)
            

            # Choose models Embeddings
            if multilingual_embeddings == "bge-m3" : 
                huggingface_embeddings = huggingface_embedding("BAAI/bge-m3", docs_after_split, )
            elif multilingual_embeddings == "multilingual-e5-large-instruct" : 
                huggingface_embeddings = huggingface_embedding("intfloat/multilingual-e5-large-instruct",docs_after_split)
            if french_embeddings == "Sentence-CamemBERT-Large" : 
                huggingface_embeddings = huggingface_embedding("dangvantuan/sentence-camembert-large", docs_after_split)
            elif french_embeddings == "FlauBERT" : 
                huggingface_embeddings = huggingface_embedding("flaubert/flaubert_base_cased", docs_after_split)
            elif french_embeddings == "BartHez" : 
                huggingface_embeddings = huggingface_embedding("moussaKam/barthez", docs_after_split)


            # Once we have a embedding model, we are ready to vectorize all our documents and store them in a vector store to construct a retrieval system.
            vectorstore = FAISS.from_documents(docs_after_split, huggingface_embeddings, distance_strategy=DistanceStrategy.COSINE)
            
            # Use similarity searching algorithm and return 3 most relevant documents.
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": kwargs})

            # Create Prompt
            prompt_template = """Utilisez les éléments de contexte suivants pour répondre à la question à la fin. Veuillez suivre les règles suivantes :
            1. Si vous ne connaissez pas la réponse, ne tentez pas d'en inventer une. Dites simplement "Je ne trouve pas la réponse finale, mais vous pouvez consulter les liens suivants".
            2. Si vous trouvez la réponse, écrivez-la de manière concise en cinq phrases maximum.

            {context}

            Question : {question}

            Réponse utile :
            """

            PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
            )

            # Call LangChain’s RetrievalQA with the prompt above.
            retrievalQA = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )

            # create a retriaval session
            if "retrivalQA" not in st.session_state:
                st.session_state.retrivalQA = retrievalQA

            # Calculate and display the number of chunks and their average size
            with st.sidebar.expander("Chunk Informations ℹ️") : 
                avg_doc_length = lambda docs: sum([len(doc.page_content) for doc in docs])//len(docs)
                avg_char_after_split = avg_doc_length(docs_after_split)
                st.write(f'Number of chunks : {len(docs_after_split)}')
                st.write(f'Average chunk size : {avg_char_after_split}')
                
    
    # if we choose a summary model
    else :  
        if model_oriented_summarization == "Roberta_BART_fixed_V1" or model_oriented_summarization == "LexLM_Longformer_BART_fixed_V1" or model_oriented_summarization == "T5_no_extraction_V1" or model_oriented_summarization == "RoBERTa_BART_dependent_V1" : 
            if "model" not in st.session_state:
                st.session_state.model = model_oriented_summarization
            # resume = llm_oriented_summarization(model_oriented_summarization,text_to_resume )
            # st.subheader("Résumé du text : ")
            # st.write(resume[0]["summary_text"])


# check whether ‘retrivalQA’, ‘llm’ or ‘model’ exists in the st.session_state object.
if "retrivalQA" in st.session_state or "llm" in st.session_state or "model" in st.session_state:

    ### chatbot

    # Check if "messages" is not in the session state; if not, initialize it as an empty list
    if "messages" not in st.session_state:
        st.session_state.messages = []


    # For each message, set the role and show the content as markdown
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):   
            st.markdown(message["content"])     


    # If the user enters a prompt, add it as a new message and display it as a chat message
    if prompt := st.chat_input("Comment puis-je vous aidez?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):      
            st.markdown(prompt)

        # Check if "retrivalQA" exists in the session state
        if "retrivalQA" in st.session_state:
            # Invoke the retrivalQA model to get the useful answer
            answer = st.session_state.retrivalQA.invoke({"query": prompt})
            prompt_template, answer = answer["result"].split("Réponse utile :")

        # Check if "llm" exists in the session state
        elif "llm" in st.session_state:
            # Invoke the llm model with the prompt to get the answer
            answer = st.session_state.llm.invoke(prompt)
                    
        # Check if "model" exists in the session state
        elif "model" in st.session_state:
            # Run the summarization function using the model with the prompt and get the summary text
            result = llm_oriented_summarization(st.session_state.model, prompt)
            answer = result[0]["summary_text"]

        # Display answer
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)






 