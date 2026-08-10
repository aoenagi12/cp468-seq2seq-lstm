# Sequence-to-Sequence Modeling: Custom LSTM vs. Modern LLM Baseline

## 1. Task Description & Dataset
- **Task:** Natural Language Generation (Headline Generation)[cite: 4]
- **Dataset:** CNN/DailyMail Dataset (citable, Apache 2.0 License)[cite: 4]
- **Splits:** Train (10,000), Validation (1,000), Test (1,000)[cite: 4]

## 2. Models
1. **Custom Seq2Seq LSTM:** PyTorch implementation with attention mechanism[cite: 4].
2. **Local LLM Baseline:** Offline inference using `google/flan-t5-base` via Hugging Face Transformers.
3. **API LLM Baseline:** Cloud inference via Gemini API (`google-genai`)[cite: 3].

## 3. Environment Setup
- Python: `3.10.x` or higher[cite: 4]
- PyTorch: `>= 2.4.0`[cite: 4]

```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
pip install -r requirements.txt