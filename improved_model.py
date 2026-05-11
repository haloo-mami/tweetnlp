import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super(ImprovedModel, self).__init__()

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # BiLSTM
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # Attention layer
        self.attention = nn.Linear(hidden_dim * 2, 1)

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Final layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)

        x = self.embedding(x)
        # (batch, seq_len, embed_dim)

        lstm_out, _ = self.lstm(x)
        # (batch, seq_len, hidden_dim*2)

        # Attention scores
        attn_weights = self.attention(lstm_out)
        # (batch, seq_len, 1)

        attn_weights = torch.softmax(attn_weights, dim=1)
        # normalize across sequence

        # Context vector
        context = torch.sum(attn_weights * lstm_out, dim=1)
        # (batch, hidden_dim*2)

        x = self.dropout(context)

        out = self.fc(x)

        return out