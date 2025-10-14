# Faithful Uncertainty Tuning (FUT)
Teaching Language Models to Faithfully Express their Uncertainty.

## Installation
```
git clone https://github.com/probabll/uncertainty-tuning.git
cd uncertainty-tuning/
pip install -e .
```

## Usage

### Generate synthetic FUT training data
```
python3 unctune/scripts/generate.py --model olmo2-13b-sft
```

### Train a FUT model
```
unctune-train \
  --model olmo2-13b-sft \
  --prompt generate_train \
  --data interweave \
  --name olmo2-13b-fut-interweave \
  --learning_rate 1e-5 \
  --stratify \
  --output_dir ./models/
```

### Evaluate the faithfulness of a model (cMFG)
```
unctune-evaluate \
  --dataset popqa \
  --model olmo2-13b-fut-interweave \
  --prompt generate_train \
  --decisiveness_model gpt-5 \
  --nli_model gpt-5
```

## Datasets
The exact filtering, subsampling and preprocessing of our evaluation sets is found here:
- [PopQA](unctune/datasets/popqa.py)  
- [Natural Questions](unctune/datasets/nq.py)  
- [TriviaQA](unctune/datasets/triviaqa.py)  

## Acknowledgements
<img src="eu_flag.png" alt="EU flag" width="60" align="left" />
This project received funding from the European Union’s Horizon Europe Research and Innovation programme under Grant Agreement No. 101070631 (UTTER).