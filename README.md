# Sequence-to-Sequence Modeling: Custom LSTM vs. Local LLM Baseline

## 1. Task Description & Dataset
* **Task:** Natural Language Generation (Headline Generation / Abstractive Summarization)
* **Dataset:** CNN/DailyMail Dataset (Apache 2.0 License)
* **Data Splits:** 10,000 Train / 1,000 Validation / 1,000 Test samples

## 2. Models & Approaches
1. **Custom Seq2Seq LSTM:** PyTorch implementation utilizing an Encoder-Decoder architecture with Bahdanau Attention mechanism.
2. **Local LLM Baseline:** Modern pretrained sequence-to-sequence transformer baseline using `google/flan-t5-base` via Hugging Face `transformers` (runs locally without cloud API dependency or rate limits).

## 3. Repository Structure
```text
.
├── models/
│   └── seq2seq_lstm.py     # Encoder, Attention, and Decoder PyTorch implementations
├── train.py                # Vocabulary building and LSTM training loop
├── llm_local_baseline.py   # Local Flan-T5 baseline headline generation
├── evaluate.py             # Automatic evaluation suite (BLEU-4, ROUGE-L)
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation