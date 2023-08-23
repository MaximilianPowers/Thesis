import torch
import numpy as np

def class_accuracy(model, loader, device) -> float:
    total_correct = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        output = model(inputs)
        total_correct += torch.sum(torch.eq(torch.argmax(output, axis=1),
                                   torch.round(targets))).item()
    return total_correct / len(loader.dataset)

def compute_cosine_score(g_1, g_2, tol=1e-5):
    cosine_scores = []
    for layer_g, layer_gN in zip(g_1, g_2):
        layer_g = np.diagonal(layer_g, axis1=1, axis2=2).copy()
        layer_gN = np.diagonal(layer_gN, axis1=1, axis2=2).copy()

        similarities = []
        for metric_g, metric_gN in zip(layer_g, layer_gN):
            norm_g = metric_g / (np.linalg.norm(metric_g) + 1e-10)
            norm_gN = metric_gN / (np.linalg.norm(metric_gN) + 1e-10)
            if max(norm_g) < tol or max(norm_gN) < tol:
                similarities.append(1)
                continue
            similarity = np.dot(norm_g, norm_gN) / (np.linalg.norm(norm_g) * np.linalg.norm(norm_gN))
            similarities.append(similarity)
        cosine_scores.append(similarities)
    return cosine_scores

def compute_magnitude_score(g_1, g_2):
    scorings = []
    for g_left, g_right in zip(g_1, g_2):
        norm_1 = np.linalg.norm(g_left, axis=(1,2))
        norm_2 = np.linalg.norm(g_right, axis=(1,2))
        scorings.append(np.abs(norm_1 - norm_2)/np.max([norm_1, norm_2], axis=0))
    return scorings