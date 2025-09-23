def compute_lenient_em(predictions: list[str],
                       reference_lists: list[list[str]]) -> float:

    correct = 0
    for pred, refs in zip(predictions, reference_lists):
        low = pred.lower()
        if any(alias.lower() in low for alias in refs):
            correct += 1
    return 100 * correct / len(predictions)