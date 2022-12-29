# Wikianswers

This projects builds on top of the openly available data provided by the Wikimedia Foundation which exposes Wikipedia articles. I use these to feed into a summarizer using [TFBartForConditionalGeneration](https://huggingface.co/docs/transformers/model_doc/bart#transformers.TFBartForConditionalGeneration) and [TFBertForQuestionAnswering](https://huggingface.co/docs/transformers/model_doc/bert#transformers.TFBertForQuestionAnswering) models to generate summaries and answers to questions, respectively.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the program
```bash
python3 wikianswers.py
```

From there, you'll be prompted to search queries for Wikipedia articles, and ask questions that the program will attempt to address.
