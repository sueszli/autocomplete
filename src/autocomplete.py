import random
from pathlib import Path
import re
import functools
import operator
import itertools
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse

import torchtext
import torchtext.vocab
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import one_hot
from torchtext.data.utils import get_tokenizer
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

import warnings

warnings.filterwarnings("ignore")
torchtext.disable_torchtext_deprecation_warning()
warnings.warn(torchtext._TORCHTEXT_DEPRECATION_MSG)

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

DATA_PATH = Path.cwd() / "data" / "fake_and_real_news_dataset.csv"


def get_sentences(datapath: Path) -> pd.Series:
    sentences = pd.read_csv(str(datapath))
    sentences = sentences["title"]
    sentences = sentences.dropna()
    sentences = sentences.apply(lambda x: re.sub("[^a-zA-Z]", " ", x))
    sentences = sentences.apply(lambda x: re.sub(r"\b[b-hj-z]\b", " ", x))  # floating single letters except 'a', 'i'
    return sentences


def get_ngram_encoding(tokenized_sentences: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    # get ngrams of tokenized sentences and encode them to vocabulary indices

    def _get_ngrams(tokenized_sentence: List[str]) -> List[List[str]]:
        # example: "hello beautiful world" -> ["hello", "hello beautiful", "hello beautiful world"]
        list_ngrams = []
        for i in range(1, len(tokenized_sentence)):
            ngram_sequence = tokenized_sentence[: i + 1]
            list_ngrams.append(ngram_sequence)
        return list_ngrams

    def _insert_oovs(ngram: List[str]) -> List[str]:
        # replace 10% of tokens with <oov> token to simulate the presence of out-of-vocabulary words
        for idx, _ in enumerate(ngram[:-1]):
            if random.uniform(0, 1) < 0.1:
                ngram[idx] = "<oov>"
        return ngram

    def _encode_to_vocab_idx(tokenized_sentence: List[str]) -> Optional[List[int]]:
        # encode by replacing each token with its index in the vocabulary
        tokens_list = []
        if tokenized_sentence[-1] in target_vocab.get_itos():  # get_itos() = get index to string
            for token in tokenized_sentence[:-1]:
                num_token = features_vocab[token] if token in features_vocab.get_itos() else features_vocab["<oov>"]  # features from features_vocab
                tokens_list.append(num_token)
            num_token = target_vocab[tokenized_sentence[-1]]  # target (last token) from target_vocab
            tokens_list.append(num_token)
            return tokens_list

        # target token is not in the vocabulary
        return None

    flatmap = lambda f, items: functools.reduce(operator.iconcat, map(f, items), [])

    # get ngram encodings
    ngrams_list = flatmap(_get_ngrams, tokenized_sentences)
    ngrams_list_oov = list(map(_insert_oovs, ngrams_list))
    input_sequences = list(filter(None, map(_encode_to_vocab_idx, ngrams_list_oov)))

    # split into feature and target lists
    X = [seq[:-1] for seq in input_sequences]
    y = [seq[-1] for seq in input_sequences]

    # pad all feature sequences to the same length
    maxlen = max(len(x) for x in X)
    X = [F.pad(torch.tensor(x), (maxlen - len(x), 0), value=0) for x in X]

    X = torch.stack(X)
    y = torch.tensor(y)
    return X, y


class Autocompleter(nn.Module):
    def __init__(self, features_vocab_total_words, target_vocab_total_words, embedding_dim, hidden_dim, dropout_prob):
        super(Autocompleter, self).__init__()

        self.embedding = nn.Embedding(features_vocab_total_words, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim, target_vocab_total_words)

    def forward(self, x):
        x = x.to(self.embedding.weight.device)
        embedded = self.embedding(x)  # convert indices to dense vectors
        lstm_out, _ = self.lstm(embedded)  # LSTM layer
        lstm_out = self.dropout(lstm_out)  # dropout layer: randomly zero out elements of the input tensor with probability p
        output = self.fc(lstm_out[:, -1, :])  # fully connected layer: map the output of the LSTM layer to the output vocabulary
        return output


def get_train_test_loader(X: torch.Tensor, y: torch.Tensor, batch_size: int, train_size: float) -> Tuple[DataLoader, DataLoader]:
    y_one_hot = one_hot(y, num_classes=len(target_vocab))
    data = TensorDataset(X, y_one_hot)

    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size

    train_data, test_data = random_split(data, [train_size, test_size])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_prediction_accuracy(model, data_loader, k=3):
    model.eval()  # eval mode
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # move to device
            output = model(batch_x)  # forward pass
            _, predicted_indices = output.topk(k, dim=1)  # get top-k predictions

            correct_predictions += torch.any(predicted_indices == torch.argmax(batch_y, dim=1, keepdim=True), dim=1).sum().item()
            total_predictions += batch_y.size(0)

    accuracy = correct_predictions / total_predictions
    return accuracy


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--batch_size", type=int, default=32)
    args.add_argument("--train_size", type=float, default=0.8)
    args.add_argument("--hidden_dim", type=int, default=200)
    args.add_argument("--epochs", type=int, default=50)
    args.add_argument("--dropout_prob", type=float, default=0.5)
    args.add_argument("--lr", type=float, default=0.0009)
    config = args.parse_args()
    config = vars(config)

    sentences = get_sentences(DATA_PATH)
    tokenizer = get_tokenizer("basic_english")
    tokenized_sentences = sentences.apply(tokenizer)

    features_vocab = torchtext.vocab.build_vocab_from_iterator(tokenized_sentences, min_freq=2, specials=["<pad>", "<oov>"], special_first=True)
    target_vocab = torchtext.vocab.build_vocab_from_iterator(tokenized_sentences, min_freq=2)

    X, y = get_ngram_encoding(tokenized_sentences)
    print(f"encoded dataset: {X.shape=}, {y.shape=}")

    """
    define model
    """

    model = Autocompleter(
        features_vocab_total_words=len(features_vocab),
        target_vocab_total_words=len(target_vocab),
        embedding_dim=max(len(x) for x in X),
        hidden_dim=config["hidden_dim"],
        dropout_prob=config["dropout_prob"],
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    model.to(device)

    """
    train model
    """

    train_loader, test_loader = get_train_test_loader(X, y, config["batch_size"], config["train_size"])

    all_accuracies = []
    all_losses = []
    epoch_split = 2
    for epoch in tqdm(range(config["epochs"])):
        model.train()  # train mode

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.argmax(dim=1))
            loss.backward()
            optimizer.step()

        if epoch % epoch_split == 0:
            accuracy = get_prediction_accuracy(model, train_loader)
            print(f'Epoch {epoch}/{config["epochs"]}, Loss: {loss.item():.4f}, Train K-Accuracy: {accuracy * 100:.2f}%')
            all_accuracies.append(accuracy)
            all_losses.append(loss.item())

    # show training graphs
    epoch_list = [i for i in range(1, config["epochs"], epoch_split)]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    axes[0].plot(epoch_list, all_accuracies, color="#5a7da9", label="Accuracy", linewidth=3)
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy Graph")
    axes[0].grid(True)

    axes[1].plot(epoch_list, all_losses, color="#adad3b", label="Accuracy", linewidth=3)
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss Graph")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    """
    test model
    """

    accuracy = get_prediction_accuracy(model, test_loader)
    print(f"Test K-Accuracy: {accuracy * 100:.2f}%")
