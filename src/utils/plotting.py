import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, labels, title, save_path=None):
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Confusion matrix saved to {save_path}")
    
    plt.show()