import json
import numpy as np
# from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import argparse
import os
from pathlib import Path
import random
from torch.autograd import Variable
from tqdm import tqdm

class RegressionTrainData():
    def __init__(self): # , model_name="olmo2-13b-sft"):
        # if model_name == "olmo2-13b-sft":
        #     name = "allenai/OLMo-2-1124-13B-SFT"
        # elif model_name == "olmo2-13b":
        #     name = "allenai/OLMo-2-1124-13B-Instruct"
        # else:
        #     raise TypeError("Insert known model")

        # self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16)
        # self.tokenizer = AutoTokenizer.from_pretrained(name)
        
        # self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        # self.model.resize_token_embeddings(len(self.tokenizer))
        # self.model.to(device="cuda")
        # self.model.eval()
        pass

    def _load_data(self, path):
        data = self._load_file(path)
        questions, answers, targets = self._preprocess(data)
        assert len(answers) == len(targets), 'No matching length'
        assert len(answers) == len(questions), 'No matching length'
        feat_target = [{'question':questions[i], 'answer':x, 'target':targets[i]} for i,x in enumerate(answers)]
        
        return feat_target

    def _load_file(self, path):
        with open(path) as f:
            d = json.load(f)
        return d['data']

    def _preprocess(self, list_items):
        questions = [item["question"] for item in list_items]
        answers = [item["answer"] for item in list_items]
        confidences = [item["confidence"] for item in list_items]

        # chat_prompts = [[{"role": "user", "content": q}] for q in questions] 
        # input_ids_prompts = self.tokenizer.apply_chat_template(chat_prompts, add_generation_prompt=True, return_tensors="pt", padding=True )
        
        # chat_prompts_responses = [[{"role": "user", "content": q}, {"role": "assistant", "content": answers[i]}] for i,q in enumerate(questions)]
        
        targets = np.asarray(confidences)

        return questions, answers, targets

class RegressionDataset(Dataset):
    def __init__(self, list_datapoints):
        self.list_datapoints = list_datapoints

    def __len__(self):
        return len(self.list_datapoints)

    def __getitem__(self, idx):
        a =  self.list_datapoints[idx]['answer']
        q = self.list_datapoints[idx]['question']
        y =  self.list_datapoints[idx]['target']
        return q, a, y

class OLMORegressor(nn.Module):
    def __init__(self, base_model, tokenizer, input_dim=100279, hidden_dim1=256, hidden_dim2=256, output_dim=1):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.regression_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim))

    def forward(self, answer, question): #, attention_mask=None):
        chat_prompts = [[{"role": "user", "content": q}] for q in question] 
        
        chat_prompts_responses = [[{"role": "user", "content": q}, {"role": "assistant", "content": answer[i]}] for i,q in enumerate(question)]
    
        input_ids_prompts = self.tokenizer.apply_chat_template(chat_prompts, tokenize=True, add_generation_prompt=False, return_tensors="pt", padding=True ) 
        # not sure if add_generation_prompt true or false, since we do not intend to generate

        input_ids_prompts_responses = self.tokenizer.apply_chat_template(chat_prompts_responses, tokenize=True, add_generation_prompt=False, return_tensors="pt", padding=True )
        input_ids_prompts_responses = input_ids_prompts_responses.clone()
        input_ids_prompts_responses = input_ids_prompts_responses.to('cuda')

        with torch.no_grad(): #should this be like that or not?
            outputs = self.base_model(input_ids=input_ids_prompts_responses, pad_token_id=self.tokenizer.pad_token_id)
        len_prompts = torch.sum(torch.where(input_ids_prompts== self.tokenizer.pad_token_id, 0, 1),1)
        len_prompt_responses = torch.sum(torch.where(input_ids_prompts_responses == self.tokenizer.pad_token_id, 0, 1),1)

        for i in range(len(answer)): #batch size
            relevant_outputs = outputs.logits[i][len_prompts[i]:len_prompt_responses[i]]#.to(device="cpu")
            features = torch.mean(relevant_outputs, 0)
            pred = self.regression_head(features)
            
            if i == 0:
                preds = pred.clone()
            else:
                preds = torch.cat([preds, pred], dim=0)

        return preds

def train_dev_split(data, validation_ratio = 0.1, stratify = False, random_state = 0):
    """One question might appear multiple times in the fine-tuning set. 
    Our custom splitting function ensures that a question (and its responses) 
    appear either in the train or validation set"""
    random.seed(random_state)

    questions = [item['question'] for item in data]
    unique_questions = list(set(questions))
    random.shuffle(unique_questions)

    split_index = len(unique_questions) - round(len(unique_questions)*validation_ratio)
    # unique_question_input_ids = [list(item) for item in unique_question_input_ids]
    train_questions = unique_questions[:split_index]
    dev_questions = unique_questions[split_index:]

    train_data=[]
    dev_data=[]
    for item in data:
        if item['question'] in dev_questions:
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
            items_bin = np.array([x['target'] for x in train_data if (x['target']>bins[i]) and (x['target']<=bins[i+1])])
            ratio_bin = len(items_bin)/len(train_data)
            print("Bin: (",bins[i],",",bins[i+1],") \n Percentage of training data: ",ratio_bin, " \n Actual datapoints: ", len(items_bin))

        n_per_bin = len(train_data) // 10
        # print('n_per_bin', n_per_bin)
        stratified_data = []
        for i in range(len(bins)-1):
            indeces_bin = np.array([j for j,x in enumerate(train_data) if (x['target']>bins[i]) and (x['target']<=bins[i+1])])
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
    parser.add_argument("--sample_batch_size", "-bs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--validation_ratio", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=10)
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

    regression_inputs_file = output_dir / f"{args.model}-{args.dataset}_text_inputs_targets.json"
    ### Create the data necessary for regression. From each datapoint, we create an input feauture and regression target pairs.
    if not regression_inputs_file.exists() or args.override_files:
        print("No regression inputs found.")

        regr_data = RegressionTrainData()

        for root, dirs, files in os.walk(data_dir):
            for f in files:
                file_name_starts = f"{args.model}-{args.dataset}-generate_train.training_data-{args.nli_model}-"
                if f.startswith(file_name_starts):
                    number_range = f.replace("file_name_starts","").replace(".json","")
                    text_inputs_targets = output_dir / f"text_input_targets_{number_range}.json"
                    if not text_inputs_targets.exists():
                        qa_pairs_target = regr_data._load_data(os.path.join(data_dir, f))  
                        with open(text_inputs_targets, 'w') as f:
                            json.dump(qa_pairs_target, f)                      
                
        inputs_targets = []
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                file_name_starts = "text_input_targets_"
                if f.startswith(file_name_starts):
                    path_f = os.path.join(output_dir, f)
                    with open(path_f) as h:
                        inputs_targets_partial = json.load(h)
                    inputs_targets = inputs_targets + inputs_targets_partial

        print(len(inputs_targets))
        with open(regression_inputs_file, 'w') as f:
            json.dump(inputs_targets, f)      

    regression_model_file = output_dir / f"{args.model}-{args.dataset}_regression_model_num.pt"
    regression_losses_file = output_dir / f"{args.model}-{args.dataset}_regression_model_num_losses.json"
    ### Train a neural network regression model
    if not regression_model_file.exists() or args.override_files:
        print("No regression model found.")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        with open(regression_inputs_file) as f:
            qa_targets = json.load(f)
        
        #Subsample to run a test regression model
        # qa_targets = random.sample(qa_targets, 2000)

        train_data, dev_data = train_dev_split(qa_targets, stratify=True)
        print('Number of train datapoints', len(train_data))
        print('Number of validation datapoints', len(dev_data))

        train_data = RegressionDataset(train_data)
        dev_data = RegressionDataset(dev_data)

        print('Example of a training datapoint')
        print(train_data[0])

        train_dataloader = DataLoader(train_data, batch_size=args.sample_batch_size, shuffle=True)
        dev_dataloader = DataLoader(dev_data, batch_size=args.sample_batch_size, shuffle=True)

        # bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        # lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "o_proj", "k_proj", "v_proj"],task_type="FEATURE_EXTRACTION")

        if args.model == "olmo2-13b-sft":
            name = "allenai/OLMo-2-1124-13B-SFT"
        elif args.model == "olmo2-13b":
            name = "allenai/OLMo-2-1124-13B-Instruct"
        else:
            raise TypeError("Insert known model")

        tokenizer = AutoTokenizer.from_pretrained(name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        
        base_model = AutoModelForCausalLM.from_pretrained(name)#, quantization_config=bnb_config, device_map={"":0})
        base_model.resize_token_embeddings(len(tokenizer))
        base_model.to(device=device)

        model = OLMORegressor(base_model=base_model, tokenizer=tokenizer, input_dim=100279).cuda()
        
        print(model)
        print(model.parameters())

        mse_loss = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.AdamW(model.parameters(), lr = args.learning_rate, weight_decay = 1e-1)

        # X_train = Variable(X_train, requires_grad=True).cuda().float()
        # y_train = Variable(y_train, requires_grad=True).cuda().float()
        # X_val = Variable(X_val, requires_grad=True).cuda().float()
        # y_val = Variable(y_val, requires_grad=True).cuda().float()

        total_val_loss = []
        total_val_cmfg = []
        total_train_loss = []
        trigger_times = 0

        for epoch in range(args.num_epochs):
            # forward propagation
            model.train()

            batch_train_loss = []
            for batch in tqdm(train_dataloader):
                q = batch[0]
                a = batch[1]
                y = batch[2].float()
                y = y.to("cuda")

                y_pred = model(a, q)
                loss = mse_loss(y_pred, y)
                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #if epoch % 2 == 0:
                batch_train_loss.append(loss.item())
            
            # if epoch % 2 == 0:
            epoch_train_loss = sum(batch_train_loss)/len(batch_train_loss)
            total_train_loss.append((epoch, epoch_train_loss))

            # if epoch % 2 == 0:
            model.eval()

            # batch_train_loss = []
            batch_val_loss = []
            batch_val_cmfg = []
            for batch in dev_dataloader:
                with torch.no_grad():
                    q = batch[0]
                    X = batch[1]
                    y_val = batch[2].float()
                    y_val = y.to("cuda")

                    y_pred_val = model(X, q)

                    val_loss = mse_loss(y_pred_val, y_val)
                    batch_val_loss.append(val_loss.item())
                    batch_val_cmfg.append(torch.mean(torch.abs(y_pred_val-y_val)).item())
                    # total_val_loss.append((epoch, val_loss.item()))
                    # total_train_loss.append((epoch, train_loss.item()))

            epoch_val_cmfg = sum(batch_val_cmfg)/len(batch_val_cmfg)
            epoch_val_loss = sum(batch_val_loss)/len(batch_val_loss)
            total_val_loss.append((epoch, epoch_val_loss))
            total_val_cmfg.append((epoch, epoch_val_cmfg))

            
            if epoch == 0:
                best_val_loss = epoch_val_loss
                best_model_state = model.state_dict()
            elif epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_state = model.state_dict()
            else:
                trigger_times += 1
                if trigger_times >= args.patience:
                    print("Early stopping triggered!")
                    break
    
            print('Epoch [{}/{}], Train Loss: {:.5f}, Validation Loss: {:.5f}, Validation cMFG: {:.5f}'.format(epoch, args.num_epochs, epoch_train_loss, epoch_val_loss, epoch_val_cmfg))
        
        print('\nTraining Complete')
        with open(regression_model_file, "wb") as f:
            torch.save(best_model_state, f)

        dict_losses = { "ValidationLoss": total_val_loss, 
                        "TrainingLoss": total_train_loss, 
                        "ValidationCMFG": total_val_cmfg,
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
        
        # if args.norm_target:
        #     dict_losses["dict_norm_target"]=dict_norm_target
        
        # if args.norm_inputs:
        #     dict_losses["dict_norm_input"]=dict_norm_input
         
        with open(regression_losses_file, 'w') as f:
            json.dump(dict_losses, f, ensure_ascii=False)

if __name__ == "__main__":
    main()

        