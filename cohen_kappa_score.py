import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Load the Excel file
file_path = "Sentiment Annotated Corpus.xlsx"
df = pd.read_excel(file_path)

# Extract annotation scores columns
annotator1 = pd.to_numeric(df.iloc[:, 8], errors='coerce')
annotator2 = pd.to_numeric(df.iloc[:, 9], errors='coerce')

# Drop rows where either annotator has invalid/missing values
valid_indices = annotator1.notna() & annotator2.notna()
annotator1_clean = annotator1[valid_indices]
annotator2_clean = annotator2[valid_indices]

# Compute Cohen's kappa
kappa = cohen_kappa_score(annotator1_clean, annotator2_clean)

print(f"Cohen's kappa between Annotator1 and Annotator2 is: {kappa*100:.2f}%")