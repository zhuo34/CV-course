import torch


def logit2label(logits):
    """
    Calculate labels from logits.
    :param logits: logits
    :return: labels
    """
    labels = torch.argmax(logits, dim=-1)
    return labels


def cal_acc_from_logits(logits, label):
    """
    Calculate classification accuracy.
    :param logits: logits
    :param label: labels
    :return: accuracy
    """
    labels = logit2label(logits)
    for _ in range(labels.ndim-1):
        label = torch.unsqueeze(label, dim=0)
    acc = torch.mean((labels == label).float())
    return acc