import os
import random
import collections
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.seq2seq_lstm import Encoder, Attention, Decoder, Seq2SeqLSTM

# Special tokens
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3


class Vocabulary:
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.w2i = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX, SOS_TOKEN: SOS_IDX, EOS_TOKEN: EOS_IDX}
        self.i2w = {v: k for k, v in self.w2i.items()}

    def build_vocab(self, texts):
        counter = collections.Counter()
        for text in texts:
            tokens = str(text).lower().split()
            counter.update(tokens)
        
        most_common = counter.most_common(self.max_size - len(self.w2i))
        for word, _ in most_common:
            if word not in self.w2i:
                idx = len(self.w2i)
                self.w2i[word] = idx
                self.i2w[idx] = word

    def encode(self, text, max_len=100, add_special_tokens=True):
        tokens = str(text).lower().split()[:max_len]
        ids = [self.w2i.get(token, UNK_IDX) for token in tokens]
        if add_special_tokens:
            ids = [SOS_IDX] + ids + [EOS_IDX]
        return ids

    def __len__(self):
        return len(self.w2i)


class Seq2SeqDataset(Dataset):
    def __init__(self, csv_path, src_vocab, trg_vocab, max_src_len=100, max_trg_len=30):
        df = pd.read_csv(csv_path)
        self.sources = df['article'].fillna("").tolist()
        self.targets = df['highlights'].fillna("").tolist()
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_src_len = max_src_len
        self.max_trg_len = max_trg_len

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        src_ids = self.src_vocab.encode(self.sources[idx], max_len=self.max_src_len, add_special_tokens=False)
        trg_ids = self.trg_vocab.encode(self.targets[idx], max_len=self.max_trg_len, add_special_tokens=True)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(trg_ids, dtype=torch.long)


def collate_fn(batch):
    src_list, trg_list = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=PAD_IDX)
    trg_padded = torch.nn.utils.rnn.pad_sequence(trg_list, batch_first=True, padding_value=PAD_IDX)
    return src_padded, trg_padded


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build Vocabularies from training data
    train_df = pd.read_csv("data/train.csv")
    src_vocab = Vocabulary(max_size=10000)
    trg_vocab = Vocabulary(max_size=8000)

    print("Building vocabularies...")
    src_vocab.build_vocab(train_df['article'])
    trg_vocab.build_vocab(train_df['highlights'])
    print(f"Source Vocab Size: {len(src_vocab)} | Target Vocab Size: {len(trg_vocab)}")

    # Create Datasets and DataLoaders
    train_dataset = Seq2SeqDataset("data/train.csv", src_vocab, trg_vocab)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # Model Hyperparameters
    EMB_DIM = 256
    HIDDEN_DIM = 512
    EPOCHS = 5
    LR = 0.001

    enc = Encoder(len(src_vocab), EMB_DIM, HIDDEN_DIM)
    attn = Attention(HIDDEN_DIM)
    dec = Decoder(len(trg_vocab), EMB_DIM, HIDDEN_DIM, attn)
    model = Seq2SeqLSTM(enc, dec, PAD_IDX, device).to(device)

    print(f"Total Trainable LSTM Parameters: {count_parameters(model):,}")

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # Training Loop
    model.train()
    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0
        for batch_idx, (src, trg) in enumerate(train_loader):
            src, trg = src.to(device), trg.to(device)

            optimizer.zero_grad()
            output = model(src, trg)

            output_dim = output.shape[-1]
            loss = criterion(output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch}/{EPOCHS}] Step [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"--- Epoch {epoch} Complete | Average Loss: {avg_loss:.4f} ---")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/lstm_seq2seq.pt")
    print("Saved model checkpoint to checkpoints/lstm_seq2seq.pt")


if __name__ == "__main__":
    main()