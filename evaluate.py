import nltk

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

nltk.download('punkt', quiet=True)

def evaluate_pair(reference, candidate):
    ref_tokens = [nltk.word_tokenize(reference.lower())]
    cand_tokens = nltk.word_tokenize(candidate.lower())
    
    smooth = SmoothingFunction().method1
    bleu = sentence_bleu(ref_tokens, cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l = scorer.score(reference, candidate)['rougeL'].fmeasure
    
    return {"BLEU-4": round(bleu * 100, 2), "ROUGE-L": round(rouge_l * 100, 2)}

# Side-by-Side Analysis Examples Template (Requirement 4.3)
qualitative_samples = [
    {
        "id": 1,
        "source": "Local authorities announced new transit schedules starting next Monday.",
        "reference": "Transit Schedules Change Next Week",
        "lstm_output": "transit schedules next week week week",  # Repetition error
        "llm_output": "New Transit Schedules Begin Next Week",  # High quality
        "error_category": "Repetition / Degeneration"
    },
    {
        "id": 2,
        "source": "The quantum computing firm announced a breakthrough in error mitigation.",
        "reference": "Quantum Breakthrough Announced",
        "lstm_output": "unk breakthrough announced",           # OOV failure
        "llm_output": "Major Quantum Breakthrough Achieved in Computing", # Elaborated
        "error_category": "Out-Of-Vocabulary (OOV)"
    }
]

if __name__ == "__main__":
    print("=== Evaluation Metrics Example ===")
    ref = "Transit Schedules Change Next Week"
    cand = "New Transit Schedules Begin Next Week"
    print(f"Metrics for '{cand}':", evaluate_pair(ref, cand))
    
    print("\n=== Qualitative Examples Structured for Report ===")
    for sample in qualitative_samples:
        print(f"[{sample['id']}] Category: {sample['error_category']}")
        print(f"  Source: {sample['source']}")
        print(f"  Ref:    {sample['reference']}")
        print(f"  LSTM:   {sample['lstm_output']}")
        print(f"  LLM:    {sample['llm_output']}\n")