# Installation
# pip install --upgrade --quiet huggingface_hub
# pip install langchain-community
# pip install langchain

token = "...." # to be replaced by a hugging face token
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
import torch
torch.cuda.set_device(torch.device('cuda:0'))
torch.cuda.empty_cache()

def llm_response(template, repo_id, token, questions) : 
  prompt = PromptTemplate.from_template(template)


  llm = HuggingFaceEndpoint(
      repo_id=repo_id,  temperature=0.5, model_kwargs={"max_length":2048, "token":token}
  )
  llm_chain = LLMChain(prompt=prompt, llm=llm)

  for question in questions:
    answer = llm_chain.invoke(question)
    print(answer)
    print("----------------------------------------------------------------------------------------------")


repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
template = """Question: {question}

              Réponse: Donne une réponse uniquement en français. Donne tous les détails et cite les lois correspondantes.
           """
questions = [
              "Puis-je continuer à travailler si je suis retraité en France ?",
              "Comment puis-je mettre fin à mon bail étudiant en France ?",
              "Quel est le rôle du tuteur d'un mineur en France ?",
              "Dois-je dire que je suis enceinte lors d'un entretien en France ?",
              "Qui doit payer les frais funéraires en France ?",
              "Quels frais sont couverts en cas d'accident du travail en France ?",
              "Quand dois-je remettre ma démission en France ",
              "Où puis-je obtenir mon extrait de casier judiciaire en France ?",
              "Pourquoi devrais-je déclarer la naissance de mon enfant en France ?",
              "Mes biens deviennent-ils communs après le mariage en France ?"
            ]

llm_response(template, repo_id, token, questions)