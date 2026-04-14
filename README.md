# Assignment 07: IMDB Text Classification in Multiple Ways

## Goal

Classify IMDB movie reviews as positive or negative using several text representations.

You will compare:

- word-level integer encoding with a learned embedding,
- `gensim` word embeddings,
- a character-based model,
- different preprocessing choices,
- different ways to handle unknown or unfound words.

## What you will submit

- One notebook or Python script that runs end-to-end
- Dataset loading and preprocessing code
- A word-level encoded classifier
- A `gensim` embedding classifier
- A character-based classifier
- An unknown-word handling comparison
- A model comparison table
- Follow-up answers

---

## Part A: Load IMDB and Compare Preprocessing

Load the IMDB dataset and create train, validation, and test splits.

Report:

- split sizes
- 2 positive and 2 negative examples
- average review length
- class balance

Create at least two preprocessing variants, for example:

- lowercase, remove simple punctuation, split on whitespace
- tokenize with spaCy or another tokenizer, lowercase, optionally remove stopwords or lemmatize

For each variant, report:

- 5 tokenized examples
- vocabulary size before cutoff
- 10 most frequent tokens
- 10 rare tokens

Write 3-5 sentences explaining which preprocessing choices you expect to help or hurt sentiment classification.

---

## Part B: Word-Level Integer Encoding Model

Build a vocabulary from the training split only.

Include:

```text
<PAD>
<UNK>
```

Choose and report:

- `min_freq`
- maximum sequence length
- vocabulary size
- padding and truncation rule

Train a classifier:

```text
token ids -> embedding -> RNN/GRU/LSTM or mean pooling -> classifier
```

Suggested model:

```python
class WordClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.GRU(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        packed_output, h_n = self.rnn(packed)
        last_hidden = h_n[-1]
        return self.fc(last_hidden)
```

Report train, validation, and test accuracy, plus 3 correct and 3 incorrect predictions.

---

## Part C: Gensim Word Embedding Model

Use `gensim` embeddings in one of these ways:

- train `gensim.models.Word2Vec` on your training reviews
- use pretrained vectors through `gensim.downloader`, if available

Create one vector per review using mean pooling, max pooling, or mean + max concatenation.

Train a classifier such as logistic regression or a small MLP.

Implement and compare at least two ways to handle words missing from the embedding vocabulary:

- map missing words to an `<UNK>` vector
- skip missing words
- use one reused random vector
- use the average of known vectors
- use a character-based fallback representation

Report found token count, missing token count, OOV strategy, validation accuracy, and test accuracy.

---

## Part D: Character-Based Classification

Build a character vocabulary from the training reviews.

Include:

```text
<PAD>
<UNK>
```

Train a character-based classifier:

```text
character ids -> character embedding -> RNN/GRU/LSTM or CNN -> classifier
```

Report:

- maximum character length
- character vocabulary size
- train accuracy
- validation accuracy
- test accuracy
- 3 examples where the character model behaves differently from the word model

---

## Part E: Compare All Approaches

Fill this table:

| Model / Representation | Preprocessing | OOV Strategy | Vocab Size | Max Length | Train Acc | Val Acc | Test Acc | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Word IDs + learned embedding | | | | | | | | |
| Gensim embeddings | | | | | | | | |
| Character-based model | | | | | | | | |

Answer:

1. Which representation worked best?
2. Which preprocessing choice mattered most?
3. Which unknown-word strategy worked best?
4. Which model would you choose if you expected many misspellings or rare words?

---

## Follow-Up Questions for Understanding

Answer:

1. What is the difference between a token, a vocabulary item, and an embedding vector?
2. Why should the vocabulary be built only from the training data?
3. What does `<UNK>` represent?
4. Why can skipping unknown words be dangerous for sentiment classification?
5. Why might a character-based model handle unseen words better than a word-level model?
6. What information can be lost when representing a review by the mean of its word vectors?
7. Why can stopword removal sometimes hurt sentiment classification?
8. Which approach was most sensitive to preprocessing in your experiments?
