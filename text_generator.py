import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# Load Dataset
with open("data/shakespeare.txt", "r", encoding="utf-8") as file:
    text = file.read().lower()

# The original Shakespeare dataset is very large and requires high memory to process
# Since my laptop has limited RAM i am using a smaller subset of the dataset

text = text[:20000]


# Text Preprocessing Character Level

chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}

vocab_size = len(chars)
print("Total vocabulary size:", vocab_size)

seq_length = 40
input_sequences = []
targets = []

for i in range(len(text) - seq_length):
    seq = text[i:i+seq_length]
    target = text[i+seq_length]

    input_sequences.append([char_to_idx[c] for c in seq])
    targets.append(char_to_idx[target])

X = torch.tensor(input_sequences, dtype=torch.long)
y = torch.tensor(targets, dtype=torch.long)

print("Preprocessing completed")


# Model LSTM

class TextGenerator(nn.Module):

    def __init__(self, vocab_size):
        super(TextGenerator, self).__init__()

        self.embedding = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128, 256, batch_first=True)
        self.fc = nn.Linear(256, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


model = TextGenerator(vocab_size)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

print(model)


# Now We Can Train Model

epochs = 10
print("Training started...")

for epoch in range(epochs):

    optimizer.zero_grad()

    output = model(X)

    loss = loss_fn(output, y)

    loss.backward()

    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")

print("Training completed")


# Text Generation Function

def generate_text(seed_text, next_chars):

    model.eval()

    generated = seed_text

    input_seq = torch.tensor([[char_to_idx[c] for c in seed_text]], dtype=torch.long)

    for _ in range(next_chars):

        output = model(input_seq)

        predicted_idx = torch.argmax(output).item()

        predicted_char = idx_to_char[predicted_idx]

        generated += predicted_char

        new_seq = generated[-seq_length:]

        input_seq = torch.tensor([[char_to_idx[c] for c in new_seq]], dtype=torch.long)

    return generated


# Generate Sample Text

print("\nGenerated Text Samples:\n")

seed = "to be or not to be "
print("Seed:", seed)

generated = generate_text(seed, 200)

print("Generated:", generated)