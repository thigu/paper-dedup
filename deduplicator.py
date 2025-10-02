import pandas as pd
import re
from pathlib import Path
from difflib import SequenceMatcher

# Load dataset
path = Path('normalized_records_quoted.csv')
df = pd.read_csv(path, quoting=0)  # quoting 0 to guess quoting; quoting minimal; we just read

# Preprocess title: normalize (lowercase, remove punctuation and extra whitespace)
def normalize_title(title):
    title = str(title)
    # lower case
    title = title.lower()
    # remove punctuation
    title = re.sub(r'[\W_]+', ' ', title)  # replace non-word/underscore with space
    # remove extra whitespace
    title = re.sub(r'\s+', ' ', title).strip()
    return title

df['normalized_title'] = df['title'].apply(normalize_title)

# High confidence duplicates: group by normalized_title
counts = df['normalized_title'].value_counts()
duplicate_titles = counts[counts > 1].index

# Mark high confidence duplicates
df['high_confidence_duplicate'] = df['normalized_title'].isin(duplicate_titles)

# Potential duplicates: for rows that are not high_confidence_duplicate
non_dup_df = df[~df['high_confidence_duplicate']]

# We'll map title index to list of potential duplicates
# We'll compute pairwise similarity of normalized titles across non duplicates
non_titles = non_dup_df['normalized_title'].tolist()

potential_flags = [False] * len(non_titles)

# Precompute indices mapping to original row index
non_indices = non_dup_df.index.tolist()

# We'll check each pair once; complexity ~ n(n-1)/2 ~ 600k pairs; within 1165; but our non duplicates maybe 1000; still ~500k pairs; manageable.
threshold = 0.85
n = len(non_titles)
for i in range(n):
    for j in range(i+1, n):
        t1 = non_titles[i]
        t2 = non_titles[j]
        # Quick length filter: only compute if difference in length ratio < 0.3
        len1, len2 = len(t1), len(t2)
        # compute similarity threshold; use SequenceMatcher ratio
        if abs(len1 - len2) / max(len1, len2) > 0.3:
            continue
        ratio = SequenceMatcher(None, t1, t2).ratio()
        if ratio >= threshold:
            potential_flags[i] = True
            potential_flags[j] = True

# Map potential flags back to DataFrame index
for idx, flag in zip(non_indices, potential_flags):
    df.loc[idx, 'potential_duplicate'] = flag

# For high confidence duplicates, potential_duplicate remains default False; set to False for clarity
df['potential_duplicate'] = df['potential_duplicate'].fillna(False)

# Count high confidence duplicates (rows flagged)
high_conf_count = df['high_confidence_duplicate'].sum()
potential_dup = df['potential_duplicate'].sum()

# Save new file
output_path = Path('normalized_records_duplicates_marked.csv')
df.drop(columns=['normalized_title']).to_csv(output_path, index=False, quoting=1)  # quoting=1 is QUOTE_ALL
print('High confidence duplicates:', high_conf_count)
print('Potential duplicates: ', potential_dup)
print('Saved to', output_path)