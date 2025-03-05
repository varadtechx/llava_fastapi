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
plt.savefig('confusion_matrix.png')
plt.close()

# Compute precision and recall
precision = precision_score(df['is_nsfw'], df['detected_nsfw'])
recall = recall_score(df['is_nsfw'], df['detected_nsfw'])

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')