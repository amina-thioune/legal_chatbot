from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch


model_name = "Equall/Saul-Instruct-v1"  # model's name
HF_TOKEN = " " # to be replaced by a hugging face token


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
    elif method =="All":
        quantization_config = None  # No quantization if method is not specified

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device,
                                                quantization_config=quantization_config, token=HF_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # Ensure tokenizer has a pad token, defaulting to EOS token if not

    return model, tokenizer, device  # Return the loaded model, tokenizer, and device