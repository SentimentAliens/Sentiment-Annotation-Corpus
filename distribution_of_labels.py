import pandas as pd
import numpy as np

# Load the Excel file, skip the legend rows
df = pd.read_excel('Sentiment Annotated Corpus.xlsx', sheet_name='Sheet1', skiprows=2)

# From previous output, columns are messy. Let's print shape and columns again
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())

# The data starts with score, id, url, numcomments, createdutc, date, title, selftext, annotator1, annotator2
# Assume columns index: 0:score, 1:id, 2:url, 3:numcomments, 4:createdutc, 5:date, 6:title, 7:selftext, 8:annotator1, 9:annotator2

annotator1 = df.iloc[:, 8] if df.shape[1] > 8 else pd.Series()
annotator2 = df.iloc[:, 9] if df.shape[1] > 9 else pd.Series()

def parse_labels(series):
    counts = {1:0, 2:0, 3:0, 4:0, 5:0}
    total = 0
    for val in series.dropna():
        if pd.isna(val): continue
        val_str = str(val).replace(' ', ',')
        labels = []
        for part in val_str.split(','):
            try:
                lbl = int(part.strip())
                if 1 <= lbl <= 5:
                    labels.append(lbl)
            except ValueError:
                pass
        for lbl in labels:
            counts[lbl] += 1
            total += 1
    return counts, total

counts1, total1 = parse_labels(annotator1)
counts2, total2 = parse_labels(annotator2)

total_counts = {k: counts1.get(k,0) + counts2.get(k,0) for k in [1,2,3,4,5]}
grand_total = total1 + total2

print("\nCounts per annotator:")
print("Annotator1:", counts1, "total:", total1)
print("Annotator2:", counts2, "total:", total2)
print("\nGrand total counts:", total_counts)
print("Grand total annotations:", grand_total)

labels_map = {1: 'Negative', 2: 'Neutral', 3: 'Positive', 4: 'Mixed/Other', 5: 'Sarcastic'}

table_data = []
for k, v in total_counts.items():
    pct = (v / grand_total * 100) if grand_total > 0 else 0
    label = labels_map.get(k, f'Label {k}')
    table_data.append({'Sentiment Label': label, 'Count': v, 'Percentage': f'{pct:.1f}%'})

print("\nTable data:")
for row in table_data:
    print(row)