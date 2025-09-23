from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import copy

class NLIModel:
    def __init__(self, model_name = "microsoft/deberta-large-mnli"):
        """
        Checks for a weak form of entailment between two candidate answers given a question, weak entailment meaning no contradiction.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).cuda()
        self.model.eval()

    def bidirectional_contra(self, question:str, answer1:str, answer2:str):
        qa_1 = question + ' ' + answer1
        qa_2 = question + ' ' + answer2

        input = qa_1 + ' [SEP] ' + qa_2
        encoded_input = self.tokenizer.encode(input, padding=True)
        
        with torch.no_grad():
            prediction = self.model(torch.tensor(torch.tensor([encoded_input]), device='cuda'))['logits']
            predicted_label = self._extract_prediction(prediction)
            
        reverse_input = qa_2 + ' [SEP] ' + qa_1
        encoded_reverse_input = self.tokenizer.encode(reverse_input, padding=True)
        with torch.no_grad():
            reverse_prediction = self.model(torch.tensor(torch.tensor([encoded_reverse_input]), device='cuda'))['logits']
            reverse_predicted_label = self._extract_prediction(reverse_prediction)

        is_contradiction = 0  #semantically not equivalent
        if (predicted_label == True) and (reverse_predicted_label == True):
            is_contradiction = 1 #semantically equivalent
        
        return is_contradiction
    
    def _extract_prediction(self, prediction):
        #Prediction labels: "label2id": {"CONTRADICTION": 0,"NEUTRAL": 1,"ENTAILMENT": 2}
        p = torch.nn.Softmax(dim=1)
        prediction = p(prediction)
        prob_contra = prediction[0][0] 
        prob_no_contra = prediction[0][1] + prediction[0][2]

        is_contradiction = prob_contra > prob_no_contra #True if no contradiction, False if contradiction
        return is_contradiction

