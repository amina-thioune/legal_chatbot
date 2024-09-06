from summarizer import Summarizer
from transformers import pipeline


def llm_oriented_summarization(model_name, text ) : 
    HF_TOKEN = ' ' # to be replaced by a hugging face token
    extractive_model = Summarizer()

    # Optional approach to extract relevant sentences in order to reduce the size of the text to match with the context window 
    extractive_summary = extractive_model(text)

    abstractive_model = pipeline('summarization', model  = "/".join(["MikaSie", model_name]), tokenizer = "/".join(["MikaSie", model_name]), token = HF_TOKEN)

    result = abstractive_model(extractive_summary)

    return result
