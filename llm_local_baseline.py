import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("Loading local model and tokenizer...")
model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_local_llm_response(text):
    prompt = f"Summarize into a concise headline: {text}"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=30)
        
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    sample_article = "NASA's Artemis program aims to land the first woman and first person of color on the Moon."
    print("\nLocal LLM Baseline Headline:")
    print(generate_local_llm_response(sample_article))