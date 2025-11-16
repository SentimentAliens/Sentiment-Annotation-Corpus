# Sentiment Annotated Corpus

This repository contains a manually annotated sentiment corpus of Slovene Reddit posts and comments and accompanying annotation guidelines. The corpus was annotated according to a consistent 5-class sentiment scheme, based on comprehension of sentence meaning and subsequent sentiment judgment.

---

## 📘 Contents

- **Sentiment Annotated Corpus.xlsx** — The dataset containing the 501 annotated sentences and their corresponding sentiment labels from two annotators.  
- **Sentiment Annotation Guideline.pdf** — The official documentation describing annotation procedures, decision rules, and label definitions.

---

## 🧩 Overview

The sentiment annotation process consists of two steps:

1. **Comprehension** — Understanding the content and communicative intent of the sentence.  
2. **Sentiment Judgment** — Assigning one of five sentiment labels based on the perceived sentiment orientation and pragmatic tone.

---

## 🎯 Sentiment Label Scheme

Each sentence is assigned **one of five sentiment categories**:

| Label | Class Name | Description |
|--------|-------------|-------------|
| 1 | **Negative** | Expresses a negative attitude, criticism, complaint, or disapproval. |
| 2 | **Neutral** | Reports objective facts or information with no evaluative stance. |
| 3 | **Positive** | Expresses approval, satisfaction, or positive emotion. |
| 4 | **Mixed** | Contains multiple or conflicting sentiments (e.g., both positive and negative). |
| 5 | **Sarcastic** | Uses positive or neutral wording with an opposite (typically negative) implied meaning; sentiment reversal is evident from context or tone. |

---

## 💡 Annotation Principles

- **Positive**: Label statements clearly expressing approval, appreciation, or happiness.  
- **Negative**: Label complaints, discontent, or negative evaluations.  
- **Neutral**: Assign to factual or descriptive sentences with no sentiment.  
- **Mixed**: Use when a sentence conveys both positive and negative sentiments or contains multiple clauses with different polarities.  
- **Sarcastic**: Assign only when irony or sentiment reversal is explicit and contextually clear.

Annotators were instructed to ask themselves:  
> “What kind of language is the speaker using?”

---

## 📊 Inter-Annotator Agreement

To assess the reliability of sentiment annotations between the two annotators, we calculated **Cohen's kappa coefficient**, a widely used statistical measure of inter-rater agreement for categorical data. Unlike simple percent agreement, Cohen's kappa accounts for the possibility that agreement may occur by chance. The Python code is available in cohen_kappa_score.py file.

In our dataset, Cohen's kappa was computed at **0.8663 (86.63%)**, which indicates an **almost perfect level of agreement according to common interpretation guidelines**. This high kappa value demonstrates that the annotators were consistent and reliable in applying the sentiment labels, lending confidence to the quality and robustness of the annotated corpus.

---

## 🧾 Related Repository

The original text corpus used for sentiment annotation was created and processed in a separate repository:  
👉 [**Corpus Creation Repository**](https://github.com/SentimentAliens/Corpus-Creation)

That repository provides detailed documentation of the **data collection process**, **preprocessing steps**, and **corpus structure** prior to annotation.

---

## ⚙️ Intended Use

The corpus can be used for:

- Training or evaluation of **sentiment analysis models**
- **Corpus-based linguistics** and **discourse studies**
- Studying **sarcasm detection**, **subjectivity**, and **evaluative language**
- Benchmarking **NLP tools** for fine-grained sentiment classification
