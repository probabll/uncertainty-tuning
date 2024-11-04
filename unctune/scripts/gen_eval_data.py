import argparse
from unctune.models import Olmo
from unctune.datasets import PopQA

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="popqa")
    parser.add_argument("--model", type=str, default="olmo")
    args = parser.parse_args()

    # Generate the data necessary for evaluation (20 samples and a greedy decoding)
