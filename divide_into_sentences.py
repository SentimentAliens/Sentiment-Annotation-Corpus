"""
Script to divide the Sentiment Annotated Corpus into sentences.
Each row in the output will contain one sentence with its annotation tag.
"""

import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')


def load_corpus(filepath: str) -> pd.DataFrame:
    """Load the sentiment annotated corpus from Excel file."""
    # Skip the first 2 rows which contain legend and header info
    df = pd.read_excel(filepath, header=2)
    return df


def resolve_annotation(annotator1: str, annotator2: str) -> str:
    """
    Resolve the final annotation from two annotators.
    If both agree, use that value.
    If they disagree, use the first annotator's value (or implement your own logic).
    
    Annotation legend:
    1 - negative
    2 - neutral (facts)
    3 - positive
    4 - mixed
    5 - sarcastic
    """
    # Handle NaN values
    if pd.isna(annotator1) and pd.isna(annotator2):
        return "unknown"
    if pd.isna(annotator1):
        return str(annotator2).strip()
    if pd.isna(annotator2):
        return str(annotator1).strip()
    
    ann1 = str(annotator1).strip()
    ann2 = str(annotator2).strip()
    
    # If annotations are identical, use that value
    if ann1 == ann2:
        return ann1
    
    # If annotations contain multiple values (e.g., "2, 1"), 
    # the text has multiple sentences with different sentiments
    # Return the combined annotation for now
    return f"{ann1}|{ann2}"


def split_into_sentences(text: str) -> list:
    """
    Split text into sentences using NLTK's sentence tokenizer.
    Handles edge cases and cleans up the sentences.
    """
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return []
    
    # Clean up the text
    text = text.strip()
    
    # Use NLTK's sentence tokenizer
    sentences = sent_tokenize(text)
    
    # Clean up each sentence
    cleaned_sentences = []
    for sent in sentences:
        sent = sent.strip()
        if sent:  # Only add non-empty sentences
            cleaned_sentences.append(sent)
    
    return cleaned_sentences


def parse_annotation_per_sentence(annotation: str, num_sentences: int) -> list:
    """
    Parse annotations that may be comma-separated for multiple sentences.
    E.g., "2, 1" means sentence 1 has annotation 2, sentence 2 has annotation 1.
    """
    if not annotation or annotation == "unknown":
        return ["unknown"] * num_sentences
    
    # Split by comma if present
    parts = [p.strip() for p in str(annotation).split(',')]
    
    # If we have annotations for each sentence, return them
    if len(parts) == num_sentences:
        return parts
    
    # If only one annotation, apply to all sentences
    if len(parts) == 1:
        return [parts[0]] * num_sentences
    
    # If mismatch, expand or truncate as needed
    if len(parts) < num_sentences:
        # Repeat the last annotation for remaining sentences
        return parts + [parts[-1]] * (num_sentences - len(parts))
    else:
        # Truncate if we have more annotations than sentences
        return parts[:num_sentences]


def get_sentiment_label(code: str) -> str:
    """
    Convert numeric code to sentiment label.
    
    Legend:
    1 - negative
    2 - neutral (facts)
    3 - positive
    4 - mixed
    5 - sarcastic
    """
    mapping = {
        '1': 'negative',
        '2': 'neutral',
        '3': 'positive',
        '4': 'mixed',
        '5': 'sarcastic'
    }
    return mapping.get(str(code).strip(), 'unknown')


def process_corpus(input_filepath: str, output_filepath: str, include_labels: bool = True):
    """
    Process the corpus and create a new Excel file where each row is one sentence.
    
    Args:
        input_filepath: Path to the original Excel file
        output_filepath: Path for the output Excel file
        include_labels: If True, convert numeric codes to sentiment labels
    """
    # Load the corpus
    df = load_corpus(input_filepath)
    
    # Prepare list to store sentence-level data
    sentence_data = []
    
    for idx, row in df.iterrows():
        # Get the original ID for reference
        original_id = row.get('id', idx)
        
        # Combine title and selftext for full text
        title = row.get('title', '')
        selftext = row.get('selftext', '')
        
        # Get annotations
        ann1 = row.get('annotator1', '')
        ann2 = row.get('annotator2', '')
        
        # Process title sentences
        title_sentences = split_into_sentences(title)
        
        # Process selftext sentences
        selftext_sentences = split_into_sentences(selftext)
        
        # Parse annotations per sentence for selftext (where annotations typically apply)
        ann1_per_sent = parse_annotation_per_sentence(str(ann1), len(selftext_sentences) if selftext_sentences else 1)
        ann2_per_sent = parse_annotation_per_sentence(str(ann2), len(selftext_sentences) if selftext_sentences else 1)
        
        # Add title as a single entry (usually the annotation applies to the whole post)
        for i, sent in enumerate(title_sentences):
            # For title, use the first annotation value
            final_ann = resolve_annotation(
                ann1_per_sent[0] if ann1_per_sent else ann1,
                ann2_per_sent[0] if ann2_per_sent else ann2
            )
            
            sentence_data.append({
                'original_id': original_id,
                'source': 'title',
                'sentence_index': i,
                'sentence': sent,
                'annotation_code': final_ann.split('|')[0] if '|' in final_ann else final_ann,
                'annotation_label': get_sentiment_label(final_ann.split('|')[0] if '|' in final_ann else final_ann) if include_labels else None,
                'annotator1': ann1,
                'annotator2': ann2
            })
        
        # Add selftext sentences with their respective annotations
        for i, sent in enumerate(selftext_sentences):
            a1 = ann1_per_sent[i] if i < len(ann1_per_sent) else ann1_per_sent[-1] if ann1_per_sent else 'unknown'
            a2 = ann2_per_sent[i] if i < len(ann2_per_sent) else ann2_per_sent[-1] if ann2_per_sent else 'unknown'
            final_ann = resolve_annotation(a1, a2)
            
            sentence_data.append({
                'original_id': original_id,
                'source': 'selftext',
                'sentence_index': i,
                'sentence': sent,
                'annotation_code': final_ann.split('|')[0] if '|' in final_ann else final_ann,
                'annotation_label': get_sentiment_label(final_ann.split('|')[0] if '|' in final_ann else final_ann) if include_labels else None,
                'annotator1': a1,
                'annotator2': a2
            })
    
    # Create the output DataFrame
    output_df = pd.DataFrame(sentence_data)
    
    # Remove annotation_label column if not needed
    if not include_labels:
        output_df = output_df.drop(columns=['annotation_label'])
    
    # Save to Excel
    output_df.to_excel(output_filepath, index=False)
    
    print(f"Processed {len(df)} original posts into {len(output_df)} sentences")
    print(f"Output saved to: {output_filepath}")
    
    return output_df


def main():
    input_file = "Sentiment Annotated Corpus.xlsx"
    output_file = "Sentiment_Corpus_Sentences.xlsx"
    
    # Process the corpus
    result_df = process_corpus(input_file, output_file, include_labels=True)
    
    # Display sample output (columns without text to avoid Unicode issues)
    print("\nSample output (first 10 rows - summary columns):")
    sample_cols = ['original_id', 'source', 'sentence_index', 'annotation_code', 'annotation_label']
    print(result_df[sample_cols].head(10).to_string())
    
    # Display statistics
    print("\n--- Statistics ---")
    print(f"Total sentences: {len(result_df)}")
    print(f"\nSentences by source:")
    print(result_df['source'].value_counts())
    print(f"\nSentences by annotation label:")
    print(result_df['annotation_label'].value_counts())


if __name__ == "__main__":
    main()
