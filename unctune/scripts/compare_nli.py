import argparse
import json
from sklearn.metrics import confusion_matrix
import numpy as np

def main(args):
    ref_nli = json.load(open(args.reference, "r"))["data"]
    ref_by_idx = {row["idx"]: [int(pred) for sublist in row["contradictions"] for pred in sublist] for row in ref_nli}

    sys_nli = json.load(open(args.system, "r"))["data"]
    sys_by_idx = {row["idx"]: [int(pred) for sublist in row["contradictions"] for pred in sublist] for row in sys_nli}

    assert len(ref_by_idx) == len(sys_by_idx)

    y_true = []
    y_pred = []

    for idx in ref_by_idx.keys():
        y_true.extend(ref_by_idx[idx])
        y_pred.extend(sys_by_idx[idx])

    print("0: no contradiction")
    print("1: contradiction")

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    print("Predicted contradiction when no contradiction present: ", cm[0, 1])
    print("Predicted no contradiction when contradiction present: ", cm[1, 0])

    accuracy = np.trace(cm) / np.sum(cm)
    print(f"Accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", "-r", type=str, help="Reference NLI file.")
    parser.add_argument("--system", "-s", type=str, help="System NLI file to evaluate.")
    args = parser.parse_args()
    main(args)