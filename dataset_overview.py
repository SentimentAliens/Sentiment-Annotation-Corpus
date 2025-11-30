import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict

# Load the FULL annotated corpus
df = pd.read_excel('Sentiment Annotated Corpus.xlsx', sheet_name='Sheet1', skiprows=2)
print("Full dataset shape:", df.shape)

# Slovenian stopwords list (common function words)
STOPWORDS = {
    'je', 'in', 'se', 'da', 'na', 'za', 'ne', 'pa', 'sem', 'bi', 'ki', 'to', 'ali', 'so', 'si', 
    'če', 'še', 'ni', 'mi', 'me', 'po', 'iz', 'do', 'od', 's', 'o', 'v', 'a', 'ter', 'kot', 
    'ampak', 'oziroma', 'namreč', 'torej', 'tudi', 'samo', 'saj', 'le', 'sicer', 'vedno', 'nič',
    'nikoli', 'vsi', 'vsak', 'nekaj', 'nekdo', 'karkoli', 'kamor', 'kjer', 'kako', 'kaj', 'www',
    'kdaj', 'kdo', 'zakaj', 'kamor', 'kam', 'tja', 'tukaj', 'tam', 'zdaj', 'takrat', 'https',
    'danes', 'jutri', 'včeraj', 'pri', 'pred', 'zadaj', 'pod', 'nad', 'ob', 'z', 'brez',
    'vse', 'kar', 'ker', 'ga', 'ko', 'jo', 'bo', 'bi', 'čez', 'redu', 'ima', 'že', 'več', 'res'
}

# Combine title and selftext
df['full_text'] = df['title'].fillna('') + ' ' + df['selftext'].fillna('')

# Clean text function for Slovenian
def clean_text(text):
    text = re.sub(r'[^a-zA-ZšđčćžŠĐČĆŽ\s\.\!\?]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['full_text'].apply(clean_text)

# Parse sentiment labels (use first annotator's first label for document-level sentiment)
def get_primary_sentiment(annotator_str):
    if pd.isna(annotator_str):
        return None
    labels = str(annotator_str).split(',')
    try:
        first_label = int(labels[0].strip())
        return first_label
    except:
        return None

df['sentiment'] = df['annotator1'].apply(get_primary_sentiment)

# Sentiment label mapping
label_map = {1: 'Negative', 2: 'Neutral', 3: 'Positive', 4: 'Mixed/Other', 5: 'Sarcastic'}

# 1. Sentence analysis by sentiment
def get_sentences(text):
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]

sentiment_sentence_stats = defaultdict(lambda: {'total_words': 0, 'total_sentences': 0})

for idx, row in df.iterrows():
    sentiment = row['sentiment']
    if sentiment is None:
        continue
        
    text = row['clean_text']
    sents = get_sentences(text)
    
    for sent in sents:
        words = re.findall(r'\b\w+\b', sent)
        words = [w for w in words if len(w) > 1]
        word_count = len(words)
        if word_count > 0:
            sentiment_sentence_stats[sentiment]['total_words'] += word_count
            sentiment_sentence_stats[sentiment]['total_sentences'] += 1

# Calculate avg sentence length per sentiment
sentiment_avg_sentence_len = {}
sentiment_doc_counts = {}
for sent, stats in sentiment_sentence_stats.items():
    if stats['total_sentences'] > 0:
        sentiment_avg_sentence_len[sent] = round(stats['total_words'] / stats['total_sentences'], 1)
    else:
        sentiment_avg_sentence_len[sent] = 0
    sentiment_doc_counts[sent] = df[df['sentiment'] == sent].shape[0]

# 2. Document lengths (overall)
doc_word_counts = []
for text in df['clean_text']:
    words = re.findall(r'\b\w+\b', text)
    words = [w for w in words if len(w) > 1]
    doc_word_counts.append(len(words))

avg_doc_len = np.mean(doc_word_counts)
total_docs = len(df)

# 3. Most frequent words OVERALL (with stopwords filtered for display)
def get_content_words(words):
    return [w for w in words if w not in STOPWORDS and 2 <= len(w) <= 20]

all_content_words = []
for text in df['clean_text']:
    words = re.findall(r'\b\w+\b', text)
    content_words = get_content_words(words)
    all_content_words.extend(content_words)

content_word_freq = Counter(all_content_words)
top_content_words = content_word_freq.most_common(20)

# 4. Most frequent CONTENT words PER SENTIMENT (STOPWORDS REMOVED)
sentiment_content_word_freq = defaultdict(Counter)

for idx, row in df.iterrows():
    sentiment = row['sentiment']
    if sentiment is None:
        continue
        
    words = re.findall(r'\b\w+\b', row['clean_text'])
    content_words = get_content_words(words)
    sentiment_content_word_freq[sentiment].update(content_words)

top_content_words_per_sentiment = {}
for sent in sentiment_content_word_freq:
    top_content_words_per_sentiment[sent] = sentiment_content_word_freq[sent].most_common(5)

# 5. Bigrams (overall) - using content words
content_bigrams = []
for text in df['clean_text']:
    words = re.findall(r'\b\w+\b', text)
    content_words = get_content_words(words)
    for i in range(len(content_words)-1):
        content_bigrams.append((content_words[i], content_words[i+1]))
content_bigram_freq = Counter(content_bigrams)
top_content_bigrams = content_bigram_freq.most_common(10)

# 6. Vocabulary (content words only)
vocab = set(all_content_words)
vocab_size = len(vocab)
total_content_tokens = len(all_content_words)
coverage = vocab_size / total_content_tokens if total_content_tokens > 0 else 0

# OVERALL sentence stats
total_sentence_words = 0
total_sentences = 0
for text in df['clean_text']:
    sents = get_sentences(text)
    for sent in sents:
        words = re.findall(r'\b\w+\b', sent)
        words = [w for w in words if len(w) > 1]
        word_count = len(words)
        if word_count > 0:
            total_sentence_words += word_count
            total_sentences += 1

avg_sentence_len = total_sentence_words / total_sentences if total_sentences > 0 else 0

# Results
results = {
    'total_documents': total_docs,
    'total_words': total_content_tokens,
    'avg_sentence_length_words': round(avg_sentence_len, 1),
    'total_sentences': total_sentences,
    'avg_document_length_words': round(avg_doc_len, 1),
    'top_20_content_words': top_content_words,
    'top_10_content_bigrams': top_content_bigrams,
    'documents_per_source': {'r/Ljubljana': total_docs},
    'content_vocabulary_size': vocab_size,
    'content_token_coverage': f"{coverage:.3f}"
}

print("\n" + "="*70)
print("FULL DATASET STATISTICS")
print("="*70)
for key, value in results.items():
    print(f"{key}: {value}")

print()
print("--- SENTENCE LENGTH ACROSS SENTIMENT LABELS ---")
for sent in [1,2,3,4,5]: 
    label = label_map.get(sent, str(sent))
    docs = sentiment_doc_counts.get(sent, 0)
    avg_len = sentiment_avg_sentence_len.get(sent, 0)
    print(f"{label:12s}: {avg_len:5.1f} words")

print()
print("--- MOST FREQUENT CONTENT WORDS PER SENTIMENT CATEGORY ---")
for sent in [1,2,3,4,5]:
    label = label_map.get(sent, str(sent))
    print(f"\n{label}:")
    for word, count in top_content_words_per_sentiment[sent]:
        print(f"  {word:12s}: {count}")
