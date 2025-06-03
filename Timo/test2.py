import pandas as pd
import matplotlib.pyplot as plt

df_train = pd.read_csv("data/arguments_training.csv")
df_test = pd.read_csv("data/arguments_test.csv")

# Normalize text (e.g., strip and lowercase)
text_columns = ["Premise", "Conclusion", "Stance"]
label_columns = [col for col in df_train.columns if col not in text_columns]

# Convert label columns to numeric
df_train[label_columns] = df_train[label_columns].apply(pd.to_numeric, errors="coerce")
df_test[label_columns] = df_test[label_columns].apply(pd.to_numeric, errors="coerce")

train_dist = df_train[label_columns].mean()
test_dist = df_test[label_columns].mean()

plt.figure(figsize=(12, 6))
x = range(len(label_columns))
plt.plot(x, train_dist, label="Train", marker='o')
plt.plot(x, test_dist, label="Test", marker='x')
plt.xticks(x, label_columns, rotation=90)
plt.ylabel("Average label frequency")
plt.title("Label Frequency Distribution (Train vs Test)")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()
