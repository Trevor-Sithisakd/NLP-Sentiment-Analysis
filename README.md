# 💬 Do They Like Me? — TikTok/Twitter Sentiment Analyser

A fun, interactive sentiment analysis app for analysing text messages and conversations. Upload a chat or type a message to find out what someone *really* thinks of you.

Built with a fine-tuned RoBERTa model trained on Twitter data, wrapped in a Streamlit UI.

---

## Features

- **Quick Message Analysis** — paste a single message and get an instant sentiment reading
- **Conversation Analysis** — upload a `.txt` file of messages (one per line) and see a full sentiment breakdown with a chart
- **Two-Party Conversation Analysis** — upload messages from two people separately and compare their sentiment side-by-side

---

## Tech Stack

| Layer | Library |
|---|---|
| ML Model | [`cardiffnlp/twitter-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment) |
| Inference | 🤗 Transformers + PyTorch |
| Text Cleaning | NLTK (stopwords, lemmatization) |
| UI | Streamlit |
| Data | Pandas |

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/nlp-sentiment-analysis.git
cd nlp-sentiment-analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

### 4. (Optional) Run the CLI

```bash
# Interactive mode
python app.py

# Batch mode — one sentence per line in your .txt file
python app.py messages.txt
```

---

## Project Structure

```
nlp-sentiment-analysis/
├── app.py                # CLI entry point (interactive + batch file mode)
├── streamlit_app.py      # Streamlit web UI
├── predictor.py          # Model loading and inference
├── cleaning.py           # Text pre-processing (URLs, mentions, stopwords, lemmatization)
└── requirements.txt
```

---

## Input Format

For file-based analysis, upload or pass a plain `.txt` file with **one message per line**:

```
I had such a great time with you today
you never listen to me
ok whatever
that was actually really funny
```

---

## Requirements

```
streamlit
transformers
torch
pandas
nltk
```

> **Note:** The model (~500MB) is downloaded from Hugging Face on first run and cached locally. Make sure you have an internet connection the first time.

---

## Limitations (v1)

- Works best on short, informal text (tweets, messages) — long formal text may give unexpected results
- Model was trained on English Twitter data; performance on other languages or writing styles will vary
- Sentiment is classified into three categories only: **Positive**, **Neutral**, **Negative**
- No emoji or slang-specific handling beyond standard tokenisation

---

## Roadmap

- [ ] Emoji sentiment support
- [ ] Confidence threshold filtering
- [ ] Export results to CSV
- [ ] Sentiment over time visualisation (for ordered conversations)
- [ ] Deploy to Streamlit Cloud

---

## Acknowledgements

- Model: [Cardiff NLP](https://github.com/cardiffnlp/tweeteval) — `twitter-roberta-base-sentiment`
- Built with [Streamlit](https://streamlit.io) and [Hugging Face Transformers](https://huggingface.co/transformers)
