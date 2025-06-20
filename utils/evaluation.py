import torch
import numpy as np
from torch.utils.data import DataLoader

def compute_ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """
    Compute the Expected Calibration Error (ECE) for a set of logits and labels.
    
    Args:
        probs (torch.Tensor): Probs after Softmax from the model (shape: [N, num_classes]).
        labels (torch.Tensor): True labels (shape: [N]).
        num_bins (int): Number of bins to use for calibration.
        
    Returns:
        float: The ECE value.
    """
    # Convert logits to probabilities
    #probs = torch.softmax(logits, dim=1)
    
    # Get predictted classes and their confidence scores
    confs, preds = probs.max(dim=1)  

    # Get correctness of predictions
    correct = (preds == labels).float()
    
    # Initialize bins
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0

    # For each bin, calculate the accuracy and confidence
    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confs >= low) & (confs < high)
        if in_bin.any():
            avg_conf = confs[in_bin].mean()
            accuracy = correct[in_bin].mean()
            weight = in_bin.float().mean() # proportion of all samples in this bin
            ece += weight * abs(avg_conf - accuracy)
    
    return ece


def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: torch.device):
    """
    Evaluate model on data loader, compute and return ECE and accuracy.
    Returns:
        tuple: (ece, accuracy)
    """
    model.eval()
    all_probs, all_labels, all_preds = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)                # (B, 3, 32, 32)
            labels = labels.to(device)                # y: (B,)
            logits = model(images)               # logits: (B, C)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)     # (,)

            all_probs.append(probs)         # list of (B, C)
            all_labels.append(labels)            # list of (B,)
            all_preds.append(preds)         # list of (B,)

            print(f"Processed {len(all_probs) * len(images)} / {len(loader.dataset)} images", end='\r')

    # Concatenate all batches
    probs  = torch.cat(all_probs, dim=0)   # (N, C)
    labels = torch.cat(all_labels, dim=0)  # (N,)
    preds  = torch.cat(all_preds, dim=0)   # (N,)

    # Compute metrics
    ece = compute_ece(probs, labels)
    accuracy = preds.eq(labels).float().mean().item()

    # Print results
    print(f"\nECE over CIFAR-10-C: {ece * 100:.2f}%")
    print(f"Accuracy over CIFAR-10-C: {accuracy * 100:.2f}%")

    return ece, accuracy
    