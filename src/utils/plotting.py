# ==========================
# Plotting Utilities
# ==========================
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, labels, title="Confusion Matrix", save_path=None, figsize=(8, 7)):
    """
    Plot and optionally save a confusion matrix heatmap.

    Args:
        cm (ndarray): Confusion matrix (2D array)
        labels (list): List of class labels (strings) for axes
        title (str): Title of the plot
        save_path (str): If provided, saves the figure to this path
        figsize (tuple): Size of the figure (default: (8, 7))
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar=True
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to {save_path}")

    plt.show()