# Uncertainty Tuning
Learning to express faithful markers of uncertainty for natural language generation models.

### Installation
```
git clone https://github.com/probabll/uncertainty-tuning.git
cd uncertainty-tuning/
pip install -e .
```

### Generating evaluation data (WIP)
After installing you can generate evaluation data as:

```
# Generates 20 samples and 1 greedy decoding on PopQA, using Olmo and the uncertainty prompt.
unctune-gen-eval-data --model olmo --data popqa --prompt uncertainty
```
