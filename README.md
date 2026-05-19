# DL- Developing a Deep Learning Model for NER using LSTM

## AIM
To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
The objective of this project is to develop a Deep Learning-based Named Entity Recognition model using Long Short-Term Memory (LSTM) networks. The model should be capable of learning contextual relationships within sequences of words and accurately predicting entity labels for each token in a sentence.

The system will take annotated text data as input, preprocess it into suitable numerical representations (such as word embeddings), and train an LSTM model to perform sequence labeling. The performance of the model will be evaluated based on metrics such as accuracy, precision, recall, and F1-score.

The developed model aims to improve entity recognition performance by leveraging the sequential learning capability of LSTMs and handling long-range dependencies in text.

<img width="530" height="712" alt="Screenshot 2026-05-19 103407" src="https://github.com/user-attachments/assets/a683f592-16e7-4eae-875a-5a21bdc07500" />


## DESIGN STEPS

### STEP 1:
Load data, create word/tag mappings, and group sentences.

### STEP 2:
Convert sentences to index sequences, pad to fixed length, and split into training/testing sets.

### STEP 3:
Define dataset and DataLoader for batching.

### STEP 4:
Build a bidirectional LSTM model for sequence tagging.

### STEP 5:
Train the model over multiple epochs, tracking loss.


## PROGRAM

### Name: Amruthavarshini Gopal

### Register Number: 212223230013

```python

class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=50, hidden_dim=100):
        super(BiLSTMTagger, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        return self.fc(x)




model = BiLSTMTagger(len(word2idx) + 1, len(tag2idx)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids)

            loss = loss_fn(
                outputs.view(-1, len(tag2idx)),
                labels.view(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_losses.append(total_loss)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids)

                loss = loss_fn(
                    outputs.view(-1, len(tag2idx)),
                    labels.view(-1)
                )

                val_loss += loss.item()

        val_losses.append(val_loss)

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss = {total_loss:.4f}, "
            f"Val Loss = {val_loss:.4f}"
        )

    return train_losses, val_losses

```
## OUTPUT

### Loss Vs Epoch Plot
<img width="578" height="77" alt="Screenshot 2026-05-19 104406" src="https://github.com/user-attachments/assets/c0adba32-07fc-46c1-a29e-5fe39380d769" />

<img width="859" height="568" alt="Screenshot 2026-05-19 104415" src="https://github.com/user-attachments/assets/1ab6cbf2-9e3e-4586-824f-b53081b773cf" />


### Sample Text Prediction
<img width="453" height="467" alt="Screenshot 2026-05-19 104428" src="https://github.com/user-attachments/assets/c120ac34-ca45-44c1-9fc7-1b3fcc764be3" />


## RESULT
Thus, an LSTM-based model for recognizing the named entities in the text has been developed successfully.
