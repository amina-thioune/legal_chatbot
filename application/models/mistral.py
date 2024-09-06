from langchain_community.llms import HuggingFaceHub


def load_mistral() :

    # Load token
    HF_TOKEN = " " # to be replaced by a hugging face token
    hf = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation",
        model_kwargs={"temperature":0.1, "max_new_tokens":500},
        huggingfacehub_api_token=HF_TOKEN)

    return hf