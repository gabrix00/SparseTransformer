import pandas as pd
import matplotlib.pyplot as plt
# Reorder the data
data_reordered = {
    'Models': ['Finetuned BERT-Base', 'Custom-BERT Without Finetuning', 'Finetuned Custom-BERT'],
    'Accuracy': [0.47, 0.15, 0.355]
}

# Create DataFrame with reordered data
df_reordered = pd.DataFrame(data_reordered)

# Plot
plt.figure(figsize=(6, 5))
plt.plot(df_reordered['Models'][1:], df_reordered['Accuracy'][1:], marker='o', linestyle='-', color='blue', label='Custom models')
plt.axhline(y=0.47, color='red', linestyle='--', label='BERT Base')

# Highlight the points
plt.scatter(df_reordered['Models'][1], df_reordered['Accuracy'][1], color='blue', zorder=3)
plt.scatter(df_reordered['Models'][2], df_reordered['Accuracy'][2], color='blue', zorder=3)

# Labels and title
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.legend()

# Set y-axis limit
plt.ylim(0, 0.5)  # Adjust the upper limit as needed


# Show plot
plt.grid(True)
plt.show()
