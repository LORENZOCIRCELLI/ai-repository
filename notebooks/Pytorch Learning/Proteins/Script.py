# %%
from Bio import SeqIO
import sys
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# %%
for record in SeqIO.parse("proteins.fasta", "fasta"):
    print(record.id)
    print(record.seq)
    print(len(record.seq))
    print()

# %%
sequences = []

for record in SeqIO.parse("proteins.fasta", "fasta"):
    sequences.append(str(record.seq))

print(len(sequences))
print(sequences[0])

# %%
data = []

for record in SeqIO.parse("proteins.fasta", "fasta"):
    data.append({
        "id":record.id,
        "sequence":str(record.seq)
    })

print(data[0])

# %%
df = pd.DataFrame(data)
df.head()

# %%
df["length"] = df["sequence"].apply(len)
df["length"].describe()

# %%
plt.hist(df["length"], bins=30)
plt.xlabel("Sequence Length")
plt.ylabel("Count")
plt.grid(alpha=0.5)
plt.show()

# %%
print(df.sequence[0])

# %%
amino_acids = "ACDEFGHIKLMNPQRSTVWY"

aa_to_int = {aa:i for i,aa in enumerate(amino_acids)}

def encode_sequence(seq):
    return [aa_to_int.get(a, 0) for a in seq]

# %%
encode_sequence("MSTNTL")

# %%
df["encoded"] = df["sequence"].apply(encode_sequence)

# %%
encoded_tensors = [torch.tensor(seq) for seq in df["encoded"]]

X = pad_sequence(encoded_tensors, batch_first=True)

# %%
X.shape

# %%
def label_protein(pid):

    if "RPL" in pid or "RPS" in pid:
        return 1
    else:
        return 0

df["label"] = df["id"].apply(label_protein)

y = torch.tensor(df["label"].values)

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# %%
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# %%
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# %%


# %%
class ProteinNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(20, 64)

        self.conv = nn.Conv1d(64, 128, 5)

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Linear(128, 2)

    def forward(self, x):

        x = self.embedding(x)

        x = x.permute(0,2,1)

        x = torch.relu(self.conv(x))

        x = self.pool(x).squeeze()

        return self.fc(x)

# %%
model = ProteinNet()

# %%
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# %%
for epoch in range(20):

    model.train()

    total_loss = 0

    for batch_X, batch_y in train_loader:

        optimizer.zero_grad()

        outputs = model(batch_X)

        loss = criterion(outputs, batch_y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print("epoch", epoch, "loss", total_loss)

# %%
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for batch_X, batch_y in test_loader:

        outputs = model(batch_X)

        predicted = torch.argmax(outputs, dim=1)

        correct += (predicted == batch_y).sum().item()

        total += batch_y.size(0)

print("Accuracy:", correct / total)

# %%
import random

random_seq = X_test[0].unsqueeze(0)

with torch.no_grad():
    pred = model(random_seq)

print(torch.softmax(pred, dim=0))

# %%
torch.save(model.state_dict(), "protein_model.pth")

# %%
def predict(sequence):

    encoded = [aa_to_int[a] for a in sequence]

    x = torch.tensor(encoded).unsqueeze(0)

    with torch.no_grad():
        pred = model(x)

    prob = torch.softmax(pred, dim=0)

    return prob

# %%
predict("MSTNTLQKLA")


