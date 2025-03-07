import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# Load CSV data from file
df = pd.read_csv('output.csv')

# Convert boolean columns to int for confusion matrix
df['is_nsfw'] = df['is_nsfw'].astype(bool).astype(int)
df['detected_nsfw'] = df['detected_nsfw'].astype(bool).astype(int)

# Compute confusion matrix
cm = confusion_matrix(df['is_nsfw'], df['detected_nsfw'])

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not NSFW', 'NSFW'], yticklabels=['Not NSFW', 'NSFW'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix_benchmark_qwen.png')
plt.close()

# Compute precision and recall
precision = precision_score(df['is_nsfw'], df['detected_nsfw'])
recall = recall_score(df['is_nsfw'], df['detected_nsfw'])

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')


import pandas as pd

# Load the CSV file
df = pd.read_csv("output.csv")

# False Positives: detected_nsfw is True but is_nsfw is False
false_positives = df[(df["detected_nsfw"] == True) & (df["is_nsfw"] == False)]

# False Negatives: detected_nsfw is False but is_nsfw is True
false_negatives = df[(df["detected_nsfw"] == False) & (df["is_nsfw"] == True)]

# Save to separate CSV files
false_positives.to_csv("false_positives.csv", index=False)
false_negatives.to_csv("false_negatives.csv", index=False)

print("False positive and false negative CSVs created successfully.")
