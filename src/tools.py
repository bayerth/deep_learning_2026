import os

os.environ["KERAS_BACKEND"] = "torch"
import keras
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class CharacterTokenizer:
    def __init__(self, text=None, vocab=None, lowercase=True):
        self.lowercase = lowercase
        self.unk_token = "<UNK>"
        self.unk_id = 0

        if vocab is None:
            if text is None:
                raise ValueError("Provide text or vocab.")
            vocab = text

        characters = "".join(
                char for char in vocab if char != self.unk_token
                )
        if self.lowercase:
            characters = characters.lower()

        self.vocab = [self.unk_token] + sorted(set(characters))
        self.char_to_id = {
                char: index for index, char in enumerate(self.vocab)
                }
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        if self.lowercase:
            text = text.lower()

        return torch.tensor(
                [self.char_to_id.get(char, self.unk_id) for char in text],
                dtype=torch.long,
                )

    def decode(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().numpy()

        token_ids = np.asarray(tokens).reshape(-1).tolist()
        return "".join(self.vocab[index] for index in token_ids)


import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, encoded_text, n_tokens=None, n_steps=50):
        self.tokens = torch.as_tensor(encoded_text, dtype=torch.long).reshape(-1)
        self.n_tokens = n_tokens
        self.n_steps = n_steps

        if len(self.tokens) <= n_steps:
            raise ValueError("Text must contain at least n_steps + 1 tokens.")

    def __len__(self):
        return len(self.tokens) - self.n_steps

    def __getitem__(self, index):
        window = self.tokens[index:index + self.n_steps + 1]
        X = F.one_hot(window[:-1], num_classes=self.n_tokens).float() if self.n_tokens is not None else window[:-1]
        y = window[1:]
        return X, y


def text_to_dataset(encoded_text, n_tokens=None, n_steps=50, batch_size=64):
    dataset = TextDataset(encoded_text, n_tokens, n_steps)
    return DataLoader(
            dataset,
            batch_size=batch_size if batch_size > 0 else None,
            shuffle=False,
            drop_last=False,
            )


def generate_text(model, tokenizer, prompt, n_next=10, one_hot=False):
    tokens = tokenizer.encode(prompt).tolist() if isinstance(tokenizer, CharacterTokenizer) \
        else tokenizer.encode(prompt).ids
    if not tokens:
        raise ValueError("The prompt must not be empty.")

    for _ in range(n_next):
        inputs = keras.ops.convert_to_tensor([tokens], dtype="int32")
        if one_hot:
            inputs = keras.ops.one_hot(inputs, num_classes=tokenizer.vocab_size)
        prediction = model.predict(inputs, verbose=0)
        next_token = int(prediction[0, -1].argmax())
        tokens.append(next_token)

    return tokenizer.decode(tokens)


def train_rnn(rnn, train_dl, eval_dl=None, epochs=20, batch_size=32, filename=None, shape=(2, 5)):
    early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=3,
            min_delta=1e-4,
            restore_best_weights=True,
            )
    history = rnn.fit(
            train_dl,
            validation_data=eval_dl,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[early_stopping]
            )
    loss, accuracy = rnn.evaluate(eval_dl, verbose=2)
    print(f"Validation loss: {loss:.4f}, Validation accuracy: {accuracy:.4f}")
    if filename is not None:
        rnn.save(filename)
    return history
