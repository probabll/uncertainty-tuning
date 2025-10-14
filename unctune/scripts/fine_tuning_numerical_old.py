import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import torch
import argparse
import os
from pathlib import Path
import random
from torch.autograd import Variable

class RegressionTrainData():
    def __init__(self, model_name="olmo2-13b-sft"):
        if model_name == "olmo2-13b-sft":
            name = "allenai/OLMo-2-1124-13B-SFT"
        elif model_name == "olmo2-13b":
            name = "allenai/OLMo-2-1124-13B-Instruct"
        else:
            raise TypeError("Insert known model")

        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        
        self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(device="cuda")
        self.model.eval()

    def _load_data(self, path):
        data = self._load_file(path)
        questions, features, targets = self._preprocess(data)
        assert len(features) == len(targets), 'No matching length'
        assert len(features) == len(questions), 'No matching length'
        feat_target = [(questions[i], x, targets[i]) for i,x in enumerate(features)]
        
        return feat_target

    def _load_file(self, path):
        with open(path) as f:
            d = json.load(f)
        return d['data']

    def _preprocess(self, list_items):
        questions = [item["question"] for item in list_items]
        answers = [item["answer"] for item in list_items]
        confidences = [item["confidence"] for item in list_items]

        chat_prompts = [[{"role": "user", "content": q}] for q in questions] 
        input_ids_prompts = self.tokenizer.apply_chat_template(chat_prompts, add_generation_prompt=True, return_tensors="pt", padding=True )
        
        chat_prompts_responses = [[{"role": "user", "content": q}, {"role": "assistant", "content": answers[i]}] for i,q in enumerate(questions)]
        
        input_ids_prompts_responses = self.tokenizer.apply_chat_template(chat_prompts_responses, add_generation_prompt=True, return_tensors="pt", padding=True )
        input_ids_prompts_responses = input_ids_prompts_responses
        
        inputs = []
        for i in range(len(input_ids_prompts_responses)):
            ids = input_ids_prompts_responses[i].to(device="cuda")
            with torch.no_grad():
                outputs = self.model(input_ids=ids.unsqueeze(0), pad_token_id=self.tokenizer.pad_token_id)
            len_prompt = torch.sum(torch.where(input_ids_prompts[i] == self.tokenizer.pad_token_id, 0, 1))
            len_prompt_response = torch.sum(torch.where(input_ids_prompts_responses[i] == self.tokenizer.pad_token_id, 0, 1))
            outputs["logits"] = outputs["logits"].to(device="cpu")
            relevant_outputs = outputs.logits[0][len_prompt:len_prompt_response]
            features = torch.mean(relevant_outputs, 0)
            inputs.append(features)

        targets = np.asarray(confidences)

        return input_ids_prompts, inputs, targets

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim1=256, hidden_dim2=256):
        super(MLPRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1),  # Output layer for regression
        )

    def forward(self, x):
        return self.model(x)

def train_dev_split(data, validation_ratio = 0.1, stratify = False, random_state = 0):
    """One question might appear multiple times in the fine-tuning set. 
    Our custom splitting function ensures that a question (and its responses) 
    appear either in the train or validation set"""
    random.seed(random_state)

    question_input_ids = [tuple(item[0].tolist()) for item in data]
    unique_question_input_ids = list(set(question_input_ids))
    random.shuffle(unique_question_input_ids)

    split_index = len(unique_question_input_ids) - round(len(unique_question_input_ids)*validation_ratio)
    unique_question_input_ids = [list(item) for item in unique_question_input_ids]
    train_questions = unique_question_input_ids[:split_index]
    dev_questions = unique_question_input_ids[split_index:]

    train_data=[]
    dev_data=[]
    for item in data:
        if item[0].tolist() in dev_questions:
            dev_data.append(item) 
        else:
            train_data.append(item)

    assert len(data) == len(train_data) + len(dev_data), 'Mismatch in lengths'
    
    if stratify == False:
        return train_data, dev_data
    else:
        print("Confidence bin distribution:")
        bins = [0.1 * i for i in range(11)]
        for i in range(len(bins)-1):
            items_bin = np.array([x[2] for x in train_data if (x[2]>bins[i]) and (x[2]<=bins[i+1])])
            ratio_bin = len(items_bin)/len(train_data)
            print("Bin: (",bins[i],",",bins[i+1],") \n Percentage of training data: ",ratio_bin, " \n Actual datapoints: ", len(items_bin))

        n_per_bin = len(train_data) // 10
        print('n_per_bin', n_per_bin)
        stratified_data = []
        for i in range(len(bins)-1):
            indeces_bin = np.array([j for j,x in enumerate(train_data) if (x[2]>bins[i]) and (x[2]<=bins[i+1])])
            indeces_samples = random.choices(indeces_bin, k=n_per_bin)
            samples = [train_data[y] for y in indeces_samples]
            stratified_data = stratified_data + samples

        random.shuffle(stratified_data)

        return stratified_data, dev_data

def normalise_targets(x, mi=None, sigma=None):
    if (mi==None) and (sigma==None):
        mi = torch.mean(x)
        sigma = torch.std(x)
    
    norm_targets = (x - mi)/sigma

    return norm_targets, mi, sigma

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="popqa_training")
    parser.add_argument("--model", "-m", type=str, default="olmo2-13b-sft") 
    parser.add_argument("--nli_model", "-nli", type=str, default="olmo2-13b")
    parser.add_argument("--sample_batch_size", "-bs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--validation_ratio", type=float, default=0.05)
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--hidden_dim1", type=int, default=512)
    parser.add_argument("--hidden_dim2", type=int, default=256)
    parser.add_argument("--override_files", action="store_true")
    parser.add_argument("--stratify", action="store_false")
    parser.add_argument("--norm_target", action="store_false")
    parser.add_argument("--norm_inputs", action="store_false")
    
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent / ".."
    data_dir = project_root / "outputs" / "train_data"
    data_dir.mkdir(exist_ok=True)

    output_dir = project_root / "outputs" / "train_numeric_data"
    output_dir.mkdir(exist_ok=True)

    regression_inputs_file = output_dir / f"{args.model}-{args.dataset}_inputs_targets.pt"
    ### Create the data necessary for regression. From each datapoint, we create an input feauture and regression target pairs.
    if not regression_inputs_file.exists() or args.override_files:
        print("No regression inputs found.")
        
        regr_data = RegressionTrainData(args.model)

        for root, dirs, files in os.walk(data_dir):
            for f in files:
                print(f)
                file_name_starts = f"{args.model}-{args.dataset}-generate_train.training_data-{args.nli_model}-"
                print(file_name_starts)
                print(f.startswith(file_name_starts))
                if f.startswith(file_name_starts):
                    number_range = f.replace("file_name_starts","").replace(".json","")
                    regression_inputs_partial_file = output_dir / f"input_targets_{number_range}.pt"
                    if not regression_inputs_partial_file.exists():
                        idsquestions_feat_target = regr_data._load_data(os.path.join(data_dir, f))                        
                        torch.save(idsquestions_feat_target, regression_inputs_partial_file)
                
        inputs_targets = []
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                file_name_starts = "input_targets_"
                if f.startswith(file_name_starts):
                    inputs_targets_partial = torch.load(os.path.join(output_dir, f))
                    inputs_targets = inputs_targets + inputs_targets_partial

        print(len(inputs_targets))
        torch.save(inputs_targets, regression_inputs_file)

    regression_model_file = output_dir / f"{args.model}-{args.dataset}_regression_model.pt"
    regression_losses_file = output_dir / f"{args.model}-{args.dataset}_regression_model_losses.json"
    ### Train a neural network regression model
    if not regression_model_file.exists() or args.override_files:
        print("No regression model found.")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        regression_input_train_file = output_dir / f"{args.model}-{args.dataset}_regression_input_train.pt"
        regression_target_train_file = output_dir / f"{args.model}-{args.dataset}_regression_target_train.pt"
        regression_input_val_file = output_dir / f"{args.model}-{args.dataset}_regression_input_val.pt"
        regression_target_val_file = output_dir / f"{args.model}-{args.dataset}_regression_target_val.pt"
        if not regression_input_train_file.exists() or not regression_target_train_file.exists() or not regression_input_val_file.exists() or not regression_target_val_file.exists() or args.override_files:
            questions_inputs_targets = torch.load(regression_inputs_file)
            train_data, dev_data = train_dev_split(questions_inputs_targets, validation_ratio = args.validation_ratio, stratify = args.stratify, random_state=args.seed)

            X_train = torch.stack([item[1] for item in train_data], dim=0)
            y_train = torch.FloatTensor([item[2] for item in train_data])

            X_val = torch.stack([item[1] for item in dev_data], dim=0)
            y_val = torch.FloatTensor([item[2] for item in dev_data])

            print(X_train.shape)
            print(X_val.shape)
            print(y_train.shape)
            print(y_val.shape)

            torch.save(X_train, regression_input_train_file)
            torch.save(y_train, regression_target_train_file)
            torch.save(X_val, regression_input_val_file)
            torch.save(y_val, regression_target_val_file)
        else:
            X_train = torch.load(regression_input_train_file)
            y_train = torch.load(regression_target_train_file)
            X_val = torch.load(regression_input_val_file)
            y_val = torch.load(regression_target_val_file)

        print(X_train.shape)
        print(X_val.shape)
        print(y_train.shape)
        print(y_val.shape)

        if args.norm_target:
            y_train, mi_train, sigma_train = normalise_targets(y_train)
            y_val, mi_val, sigma_val = normalise_targets(y_val, mi_train, sigma_train)
            dict_norm_target = {"mi":mi_train.item(), "sigma":sigma_train.item()}
        
        if args.norm_inputs:
            dict_norm_input = {}
            for dim in range(X_train.shape[1]):
                X_train[:, dim], mi_train, sigma_train = normalise_targets(X_train[:, dim])
                X_val[:, dim], mi_val, sigma_val = normalise_targets(X_val[:, dim], mi_train, sigma_train)
                dict_norm_input[dim] = {"mi":mi_train.item(), "sigma":sigma_train.item()}
                

        model = MLPRegressor(input_dim=X_train.shape[1], hidden_dim1=args.hidden_dim1, hidden_dim2=args.hidden_dim2)
        model.to(device) # send our model to gpu if available else cpu. 

        print(model)
        print(model.parameters())

        mse_loss = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.AdamW(model.parameters(), lr = args.learning_rate, weight_decay = 1e-1)

        X_train = Variable(X_train, requires_grad=True).cuda().float()
        y_train = Variable(y_train, requires_grad=True).cuda().float()
        X_val = Variable(X_val, requires_grad=True).cuda().float()
        y_val = Variable(y_val, requires_grad=True).cuda().float()

        total_val_loss = []
        total_train_loss = []

        for epoch in range(args.num_epochs):
            # forward propagation
            model.train()

            y_pred = model(X_train)
            loss = mse_loss(y_pred, y_train)
            
            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    y_pred_val = model(X_val)

                    val_loss = mse_loss(y_pred_val, y_val)
                    train_loss = mse_loss(y_pred, y_train)
                    total_val_loss.append((epoch, val_loss.item()))
                    total_train_loss.append((epoch, train_loss.item()))

                    if epoch == 0:
                        best_val_loss = val_loss.item()
                        best_model_state = model.state_dict()
                    elif val_loss < best_val_loss:
                        best_val_loss = val_loss.item()
                        trigger_times = 0
                        best_model_state = model.state_dict()
                    else:
                        trigger_times += 1
                        if trigger_times >= args.patience:
                            print("Early stopping triggered!")
                            break
    
                print('Epoch [{}/{}], Train Loss: {:.5f}, Validation Loss: {:.5f}'.format(epoch, args.num_epochs, loss.item(), val_loss.item()))
        
        print('\nTraining Complete')
        torch.save(best_model_state, regression_model_file)

        dict_losses = { "ValidationLoss": total_val_loss, 
                        "TrainingLoss": total_train_loss, 
                        "dataset":args.dataset,
                        "patience":args.patience,
                        "learning_rate":args.learning_rate,
                        "stratify":args.stratify,
                        "validation_ratio":args.validation_ratio,
                        "num_epochs":args.num_epochs,
                        "hidden_dim1":args.hidden_dim1,
                        "hidden_dim2":args.hidden_dim2,
                        "norm_target":args.norm_target,
                        "norm_inputs":args.norm_inputs}
        
        if args.norm_target:
            dict_losses["dict_norm_target"]=dict_norm_target
        
        if args.norm_inputs:
            dict_losses["dict_norm_input"]=dict_norm_input
         
        with open(regression_losses_file, 'w') as f:
            json.dump(dict_losses, f, ensure_ascii=False)

if __name__ == "__main__":
    main()

        