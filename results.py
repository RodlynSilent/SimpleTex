import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Sample data for Keyword Extraction Accuracy
manual_keywords = ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
extracted_keywords = ["keyword1", "keyword2", "keyword3", "keyword5", "keyword6"]

# Calculate true positives, false positives, and false negatives
true_positives = len(set(manual_keywords).intersection(set(extracted_keywords)))
false_positives = len(set(extracted_keywords) - set(manual_keywords))
false_negatives = len(set(manual_keywords) - set(extracted_keywords))
true_negatives = len(manual_keywords) - true_positives

# Create a confusion matrix
conf_matrix = np.array([[true_positives, false_positives],
                        [false_negatives, true_negatives]])

# Plot confusion matrix
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_xticklabels(['Relevant', 'Irrelevant'])
ax.set_yticklabels(['Relevant', 'Irrelevant'])
ax.set_title('Confusion Matrix for Keyword Extraction')
plt.show()

# Sample data for Performance
text_sizes = np.array([100, 500, 1000, 5000, 10000, 20000])  # Number of words
processing_times = np.array([0.1, 0.3, 0.5, 1.2, 2.4, 4.8])  # Time in seconds
memory_usages = np.array([50, 70, 100, 150, 200, 300])  # Memory usage in MB

# Plot Processing Time vs Text Size
plt.figure(figsize=(8, 5))
plt.plot(text_sizes, processing_times, marker='o', linestyle='-', color='b')
plt.title('Processing Time vs Text Size')
plt.xlabel('Text Size (words)')
plt.ylabel('Processing Time (seconds)')
plt.grid(True)
plt.show()

# Plot Memory Usage vs Text Size
plt.figure(figsize=(8, 5))
plt.plot(text_sizes, memory_usages, marker='o', linestyle='-', color='g')
plt.title('Memory Usage vs Text Size')
plt.xlabel('Text Size (words)')
plt.ylabel('Memory Usage (MB)')
plt.grid(True)
plt.show()
