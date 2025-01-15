from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("[INST]What is the meaning of {word}[/INST]")

llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={
        "max_new_tokens": 250,
    },
)

chain = prompt | llm

chain.invoke({"word": "potato"})