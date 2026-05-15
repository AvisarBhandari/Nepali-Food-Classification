
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def get_all_preds(model, loader, device):

    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)

            outputs = model(X)

            preds.append(outputs.argmax(dim=1).cpu().numpy())
            labels.append(y.numpy())

    return np.concatenate(preds), np.concatenate(labels)


def plot_confusion_matrix(model, loader, class_names, device):

    y_pred, y_true = get_all_preds(model, loader, device)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
def top_k_pred_accuracy(output,target,k =3):
    with torch.no_grad():
        _,pred = output.topk(k, True, True,True)
        correct = pred.ep(target.view(-1,1).expand_as(pred))
        return correct.any(dim=1).float().mean().item()
    
