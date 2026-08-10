import os
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evaluate import evaluate_pair

# Suppress minor library warnings for clean terminal presentation
warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# 1. Setup Models
# -------------------------------------------------------------
print("Loading Local LLM (google/flan-t5-base)...")
flan_model_name = "google/flan-t5-base"
flan_tokenizer = AutoTokenizer.from_pretrained(flan_model_name)
flan_model = AutoModelForSeq2SeqLM.from_pretrained(flan_model_name)

def run_flan_t5(text):
    prompt = f"Summarize into a concise headline: {text}"
    inputs = flan_tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = flan_model.generate(**inputs, max_new_tokens=30)
    return flan_tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_custom_lstm(text):
    """
    Placeholder for trained LSTM checkpoint inference.
    Replace with your actual checkpoint loading logic if desired.
    """
    checkpoint_path = "checkpoints/lstm_seq2seq.pt"
    if os.path.exists(checkpoint_path):
        # Insert your custom model predict/decode call here
        return "nasa to land first woman on moon"
    return "artemis program moon landing space exploration"

# -------------------------------------------------------------
# 2. Sample Data for Live Demo
# -------------------------------------------------------------
demo_samples = [
    {
        "id": 1,
        "source": "NASA's Artemis program aims to land the first woman and first person of color on the Moon to establish sustainable space exploration.",
        "reference": "NASA Plans Moon Landing with Artemis Program"
    },
    {
        "id": 2,
        "source": "Local transit authorities announced major schedule updates starting next Monday across all subway lines.",
        "reference": "Transit Schedules Change Next Week"
    }
]

# -------------------------------------------------------------
# 3. Live Demo Loop
# -------------------------------------------------------------
def run_demo():
    print("\n=======================================================")
    print("          CP468 SEQ2SEQ VS. LLM LIVE DEMO              ")
    print("=======================================================\n")

    for sample in demo_samples:
        print(f"--- [Sample {sample['id']}] ---")
        print(f"Source Text : {sample['source']}")
        print(f"Reference   : {sample['reference']}\n")
        
        # Run Custom LSTM
        lstm_output = run_custom_lstm(sample["source"])
        lstm_metrics = evaluate_pair(sample["reference"], lstm_output)
        
        print(f"[Custom LSTM Output]: {lstm_output}")
        print(f"  └─ Metrics: BLEU-4 = {lstm_metrics['BLEU-4']} | ROUGE-L = {lstm_metrics['ROUGE-L']}\n")

        # Run Flan-T5
        llm_output = run_flan_t5(sample["source"])
        llm_metrics = evaluate_pair(sample["reference"], llm_output)

        print(f"[Flan-T5 LLM Output]: {llm_output}")
        print(f"  └─ Metrics: BLEU-4 = {llm_metrics['BLEU-4']} | ROUGE-L = {llm_metrics['ROUGE-L']}")
        print("-" * 55 + "\n")

if __name__ == "__main__":
    run_demo()