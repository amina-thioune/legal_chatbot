# Installation
# pip install accelerate

import transformers
import torch
torch.cuda.set_device(torch.device('cuda:0'))
torch.cuda.empty_cache()
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import sys
import re


model_name = "Equall/Saul-Instruct-v1"  # model's name
HF_TOKEN = '...' # to be replaced by a hugging face token


def load_model(HF_TOKEN=HF_TOKEN, model_name=model_name, method=None):
    # Load LLM
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Check if GPU is available
    if method == "8 bits":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)  # Configure quantization for 8-bit
    elif method == "4 bits":
        quantization_config = BitsAndBytesConfig(
                      load_in_4bit=True,
                      bnb_4bit_use_double_quant=True,
                      bnb_4bit_quant_type="nf4",
                      bnb_4bit_compute_dtype=torch.bfloat16
        )  # Configure quantization for 4-bit with specific options
    else:
        quantization_config = None  # No quantization if method is not specified

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device,
                                                quantization_config=quantization_config, token=HF_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # Ensure tokenizer has a pad token, defaulting to EOS token if not

    return model, tokenizer, device  # Return the loaded model, tokenizer, and device


def extract_sentences(paragraph):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', paragraph)
    extracted_sentences = sentences[:-1]  # Split paragraph into sentences and remove the last element (empty string)

    return ' '.join(extracted_sentences)  # Return concatenated sentences as a single string


def predict(prompt, model, tokenizer, device):
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)  # Tokenize prompt and move to specified device
    generated_ids = model.generate(**model_inputs, pad_token_id=tokenizer.pad_token_id,
                                   max_new_tokens=500, do_sample=True)  # Generate text using the model
    return extract_sentences(tokenizer.batch_decode(generated_ids)[0])  # Decode generated tokens and return as concatenated sentences







if __name__ == "__main__":
    model, tokenizer, device = load_model()

    texte ="Texte à résumé"

    template = """Texte: {texte}

                Résumé : Donne un résumé court et précis en français.
                """
  
    answer = predict(template, model, tokenizer, device)
    print(f"Résumé : {answer}")
   