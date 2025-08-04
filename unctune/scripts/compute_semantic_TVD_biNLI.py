from unctune.models import GenerationConfig, OpenAIModel
# from unctune.semantic_equivalence import NliEntailLLM
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from unctune.scripts.utils import load_model

from collections import Counter
import random
import torch
import json
import time

def get_dist_probs(support, counts_support):
    """Receives the desired distribution support and a list with repeated counts for each of the elements in the support.
    e.g. support: [1,2,3,4] counts_support: [2,2,1,3,1,1,1]
    Returns the probability of each of the elements of the support given their counts.
    e.g. probs: [4/7, 2/7, 1/7, 0]"""
    total_counts = len(counts_support)
    counts_per_element = dict(Counter(counts_support))

    probs = [counts_per_element[element]/total_counts if element in list(counts_per_element.keys()) else 0 for element in support]
    return probs

def get_tvd(probs1, probs2):
    """Receives list of probabilities from 2 distributions with a shared support 
    and returns ther Total Variation Distance = 1/2 * Sum_{x} |P(x) - Q(x)|"""
    abs_differences = [abs(p - probs2[i]) for i,p in enumerate(probs1)]
    tvd = sum(abs_differences)/2
    return tvd

def nli_bidirectional_entailment(model, tokenizer, question, answer1, answer2, lenient=False):
    qa_1 = question + ' ' + answer1
    qa_2 = question + ' ' + answer2

    input = qa_1 + ' [SEP] ' + qa_2
    encoded_input = tokenizer.encode(input, padding=True)
    with torch.no_grad():
        prediction = model(torch.tensor(torch.tensor([encoded_input]), device='cuda'))['logits']
        predicted_label = torch.argmax(prediction, dim=1)

    reverse_input = qa_2 + ' [SEP] ' + qa_1
    encoded_reverse_input = tokenizer.encode(reverse_input, padding=True)
    with torch.no_grad():
        reverse_prediction = model(torch.tensor(torch.tensor([encoded_reverse_input]), device='cuda'))['logits']
        reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)

    is_equivalent = 0  #semantically not equivalent
    if lenient:
        if (2 in predicted_label) or (2 in reverse_predicted_label):
            is_equivalent = 1 #semantically equivalent
    else:
        if (2 in predicted_label) and (2 in reverse_predicted_label):
            is_equivalent = 1 #semantically equivalent
    
    return is_equivalent

def semantic_clusters(question, samples_a, samples_b, nli_model, nli_tokenizer):
    """Merges samples and clusters them semantically.
    This is done by iterating over samples. For each sample, we check whether it belongs to an existing cluster
    (by checking if it is semantically equivalent to a randomly chosen sample in the cluster). If not, the sample
    is assigned a new cluster"""
    joint_set_samples = list(set(samples_a + samples_b)) #unique responses
    
    ### Create semantic clusters
    clusters = {}
    sample_to_cluster = {}

    clusters[0] = [joint_set_samples[0]]
    sample_to_cluster[joint_set_samples[0]] = 0
    if len(joint_set_samples)>1:
        for sample in joint_set_samples[1:]:
            cluster_assignment = None
            for clus_num in clusters.keys():
                compare_sample = random.sample(clusters[clus_num],1)

                #Semantic equivalence using bidirectional entailment
                is_equivalent = nli_bidirectional_entailment(nli_model, nli_tokenizer, question, sample, compare_sample[0])
                
                if is_equivalent == 1:
                    cluster_assignment = clus_num #this is the cluster we assign to the sample
                    break
                    
            if cluster_assignment == None: 
                #if we iterated over all existing clusters, 
                #but none of them is semantically equivalent to the sample
                new_cluster = max(list(clusters.keys())) + 1
                clusters[new_cluster] = [sample]
                sample_to_cluster[sample] = new_cluster
            else: #if we found a semantically equivalent cluster
                clusters[cluster_assignment].append(sample)
                sample_to_cluster[sample] = cluster_assignment

    return clusters, sample_to_cluster

def compute_semantic_TVD(question, samples_initial, samples_final, nli_model, nli_tokenizer, print_intermediate_results=False):
    """Receives a question (string), samples from the model before fine-tuning, samples_initial (list of strings) and 
    samples from the model after fine-tuning, samples_final (list of strings).
    
    The joint set of samples is clustered semantically, the initial (pre fine-tuning) and final (post fine-tuning) semantic distributions
    are reconstructed, and their TVD is computed."""
    
    #Semantic clustering
    clusters, sample_to_cluster = semantic_clusters(question, samples_initial, samples_final, nli_model, nli_tokenizer)

    ### Compute faithfulness scores
    #Response to cluster re-expression
    clusters_initial = [sample_to_cluster[sample] for sample in samples_initial]
    clusters_final = [sample_to_cluster[sample] for sample in samples_final]            

    distribution_support = list(clusters.keys())
    distribution_support = [int(x) for x in distribution_support]

    distribution_probs_initial = get_dist_probs(distribution_support,clusters_initial)
    distribution_probs_final = get_dist_probs(distribution_support,clusters_final)

    tvd_initial_vs_final = get_tvd(distribution_probs_initial, distribution_probs_final)

    if print_intermediate_results == True:
        print(clusters)
        print(distribution_support)
        print(distribution_probs_initial)
        print(distribution_probs_final)

    return tvd_initial_vs_final, clusters, sample_to_cluster
