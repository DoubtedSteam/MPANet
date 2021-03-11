import torch


def calc_acc(logits, label, ignore_index=-100, mode="multiclass"):
    if mode == "binary":
        indices = torch.round(logits).type(label.type())
    elif mode == "multiclass":
        indices = torch.max(logits, dim=1)[1]

    if label.size() == logits.size():
        ignore = 1 - torch.round(label.sum(dim=1))
        label = torch.max(label, dim=1)[1]
    else:
        ignore = torch.eq(label, ignore_index).view(-1)

    correct = torch.eq(indices, label).view(-1)
    num_correct = torch.sum(correct)
    num_examples = logits.shape[0] - ignore.sum()

    return num_correct.float() / num_examples.float()
