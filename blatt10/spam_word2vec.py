import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

nltk.download('stopwords', quiet=True)

# Word2Vec-Encoder (Skip-Gram) von lösungen auf ilias übernommen
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_embeddings = nn.Linear(embedding_dim, vocab_size, bias=False)
        nn.init.uniform_(self.in_embeddings.weight, -0.5, 0.5)

    def forward(self, words):
        latents = self.in_embeddings(words)
        logits = self.out_embeddings(latents)
        return logits, latents

# Spam-Klassifikationsnetzwerk (mit GPT generiert)
class SpamClassifier(nn.Module):
    def __init__(self, embedding_dim):
        super(SpamClassifier, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)

# Dataset-Klasse für SMS-Daten
class SMSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, word2idx, embedding_model):
        self.data = []
        for text, label in zip(texts, labels):
            tokens = [tokenizer(t) for t in text.lower().split()]
            tokens = [word2idx[t] for t in tokens if t in word2idx]
            if tokens:
                embedding = embedding_model.in_embeddings.weight[tokens].mean(dim=0)
                self.data.append((embedding, torch.tensor([label], dtype=torch.float32)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Trainingsroutine A: Task Head
def train_task_head_only(task_head, dataloader, loss_fn, optimizer, device, epochs=5):
    task_head.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for embeddings, labels in dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = task_head(embeddings)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

# Trainingsroutine B: Encoder + Task Head
def train_encoder_and_task_head(encoder, task_head, dataloader, loss_fn, optimizer, device, epochs=5):
    encoder.train()
    task_head.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for embeddings, labels in dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            _, latent_vectors = encoder(embeddings)
            outputs = task_head(latent_vectors.mean(dim=1))
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")



def calculate_accuracy(output, labels):
    predictions = output.round() # Rundet die Ausgabe auf 0 oder 1
    correct = (predictions == labels).float() # Konvertiert in float für die␣
    ↪Division
    accuracy = correct.sum() / len(correct)
    return accuracy
