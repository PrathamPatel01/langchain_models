# from langchain_huggingface import HuggingFaceEndpoint
# from dotenv import load_dotenv
# import os

# load_dotenv()
# api_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v0.6",  # hosted model
#     task="text-generation",
#     huggingfacehub_api_token=api_key
# )

# result = llm.invoke("What is the capital of France?")
# print(result)
import torch
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v0.6",  # this one is actually chat-finetuned
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

messages = [
    {"role": "system", "content": "You are a rude chatbot."},
    {"role": "user", "content": "What is the capital of India?"}
]

prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])