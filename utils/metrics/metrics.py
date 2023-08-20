import torch


def class_accuracy(model, loader, device) -> float:
    total_correct = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        output = model(inputs)
        total_correct += torch.sum(torch.eq(torch.argmax(output, axis=1),
                                   torch.round(targets))).item()
    return total_correct / len(loader.dataset)
