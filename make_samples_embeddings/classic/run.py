import pandas as pd
import torch
import json
import torch.nn.functional as F
from transformers import LongformerTokenizerFast, LongformerModel
from gensim.models.doc2vec import Doc2Vec
from gensim.corpora.wikicorpus import tokenize
from gensim.models.keyedvectors import KeyedVectors
from utils import WikiTextCleaner
import pickle5 as pickle
import numpy as np
import argparse

def parse_args():
    """
    Text-agnostic habitat classification model
    """
    parser = argparse.ArgumentParser(description="Hyperparameters and config parser")

    parser.add_argument("--LEVEL", dest="LEVEL",
                      help="{Ecosystem classification level (L1 or L2)}",
                      default = "L1",
                      type=str)

    parser.add_argument("--RANDOM_STATE", dest="RANDOM_STATE",
                      help="{Random state used to create the dataset}",
                      default = 1,
                      type=int)

    parser.add_argument("--SPLITTING", dest="SPLITTING",
                      choices=["spatial", "species", "all"],
                      help="{data splitting criterion, can be spatial or species}",
                      default = "spatial",
                      type=str)

    parser.add_argument("--SPLIT", dest="SPLIT",
                      choices=["train", "test", "val"],
                      help="{data splitting criterion, can be spatial or species}",
                      default = "train",
                      type=str)
    args = parser.parse_args()
    return args

class SplitProcessor():
    def __init__(self):
        """Class initialization"""
        self.device = "cuda:0"
        ## Models
        self.dv_tokenizer = tokenize
        #'/scratch/izar/santacro/models/doc2vec_light.pickle'
        with open('/scratch/izar/santacro/models/doc2vec_light.pickle', 'rb') as handle:
            self.dv_model = pickle.load(handle)
        self.lf_tokenizer = LongformerTokenizerFast.from_pretrained("allenai/longformer-base-4096")
        self.lf_model = LongformerModel.from_pretrained("allenai/longformer-base-4096").to(device=self.device)
        self.lf_model.eval()

    def load_file(self, path, random_state = 1, level="L1", splitting="spatial", split="train"):
        ## Path and data
        self.path = path
        self.random_state = random_state
        self.level = level
        self.splitting = splitting
        self.split = split
        self.file_name = self.path + f"{self.random_state}_{self.level}_{self.splitting}_based_{self.split}_data.json"
        self.split_data = pd.read_json(self.path + f"{self.random_state}_{self.level}_{self.splitting}_based_{self.split}_data.json", orient="records")
        self.species_keys = pd.read_json(self.path + f"{self.random_state}_{self.level}_species_keys.json", orient="records")
        self.habitats_keys = pd.read_json(self.path + f"{self.random_state}_{self.level}_habitats_keys.json", orient="records")
        self.classes_codes = self.habitats_keys["class"]
        self.classes_names = self.habitats_keys["TypoCH_FR"].apply(lambda x : x[:20])
        self.num_classes = len(self.classes_codes)

    def get_agnostic_inputs(self, species_keys, input_size):
        """Retrieve textual inputs from json files"""
        input = torch.tensor(self.species_keys[self.species_keys["species_key"].isin(species_keys)].ID.values)
        input = F.one_hot(input, num_classes=input_size).sum(dim=0)
        return input.tolist()
    
    def get_joined_inputs(self, species_keys, embedding_method="doc2vec", fusion="sum", normalize=True, input_size=512):
        merged_embeddings = torch.zeros((len(species_keys),input_size))
        i = 0
        ## Aggregate embeddings of present species using max pooling
        for key in species_keys:
            with open(self.path + f"species/{key}.json", "r") as fp:
                file_content = json.load(fp)
            species_embeddings = file_content[embedding_method + "_embedding"]
            merged_embeddings[i,:] = torch.tensor(species_embeddings)
            i += 1
        if fusion == "max":
            merged_embeddings, _ = merged_embeddings.max(dim=0)
            if normalize:
                merged_embeddings = merged_embeddings.div(merged_embeddings.norm())
        elif fusion == "prod":
            merged_embeddings = merged_embeddings.prod(dim=0)
            if normalize:
                merged_embeddings = merged_embeddings.div(merged_embeddings.norm())
        elif fusion == "sum":
            merged_embeddings = merged_embeddings.sum(dim=0)
            if normalize:
                merged_embeddings = merged_embeddings.div(merged_embeddings.norm())
        else:
            raise Exception("invalid embeddings fusion method, must be 'max', 'prod', or 'mean'.")
        return merged_embeddings.tolist()
    
    def get_selected_doc2vec_inputs(self, species_keys):
        """Retrieve textual inputs from json files"""
        ## To store all paragraphs
        selected_pars = []
        for key in species_keys:
            with open(self.path + f"species/{key}.json", "r") as fp:
                file_content = json.load(fp)
            pars = file_content["paragraphs"]
            scores = file_content["paragraphs_eco_scores"]
            good_pars = [pars[scores.index(a)] for a in scores if a > 0.5]
            ## If some paragraphs are good, use them as inputs
            if len(good_pars) > 0:
                selected_pars += good_pars
            ## If none is sufficient, just keep the one with the highest score
            else:
                selected_pars += [pars[scores.index(max(scores))]]
        selected_pars = "\n".join(selected_pars)
        tokenized_pars = self.dv_tokenizer(selected_pars)
        embedded_pars = self.dv_model.infer_vector(tokenized_pars)
        return embedded_pars.tolist()

    def get_selected_longformer_inputs(self, species_keys, batch_size=18):
        all_examples = species_keys.to_list()
        all_embeddings = []
        for i in range(np.ceil(len(all_examples)/batch_size).astype(int)):
            batch = all_examples[i*batch_size:(i+1)*batch_size]
            batch_pars = []
            for example in batch:
                ## build batch
                selected_pars = []
                for key in example:
                    with open(self.path + f"species/{key}.json", "r") as fp:
                        file_content = json.load(fp)
                    pars = file_content["paragraphs"]
                    scores = file_content["paragraphs_eco_scores"]
                    good_pars = [pars[scores.index(a)] for a in scores if a > 0.5]
                    ## If some paragraphs are good, use them as inputs
                    if len(good_pars) > 0:
                        selected_pars += good_pars
                    ## If none is sufficient, just keep the one with the highest score
                    else:
                        selected_pars += [pars[scores.index(max(scores))]]
                selected_pars = "\n".join(selected_pars)
                batch_pars.append(selected_pars)
            tokenized = self.lf_tokenizer(batch_pars, return_tensors="pt", padding="max_length", truncation=True, max_length=4096)
            ## infer
            with torch.no_grad():
                embedded = self.lf_model(input_ids = tokenized.input_ids.to(device=self.device), attention_mask = tokenized.attention_mask.to(device=self.device))["pooler_output"].cpu()
            all_embeddings[i*batch_size:(i+1)*batch_size] = embedded.tolist()
        return pd.Series(all_embeddings)
    
    def get_all_embeddings(self):
        ## agnostic ones
        #self.split_data = self.split_data[:100]
        #print("Getting text agnostic embeddings")
        #self.split_data["agnostic_embedding"] = self.split_data["species_key"].apply(lambda x : self.get_agnostic_inputs(x,len(self.species_keys)))
        print("Getting doc2vec joined embeddings")
        self.split_data["doc2vec_joined_embedding"] = self.split_data["species_key"].apply(lambda x : self.get_joined_inputs(x,"doc2vec","max",True,768))
        #print("Getting longformer joined embeddings")
        #self.split_data["longformer_joined_embedding"] = self.split_data["species_key"].apply(lambda x : self.get_joined_inputs(x,"longformer","max",False,768))
        #self.split_data["longformer_joined_embedding"] = self.split_data["species_key"].apply(lambda x : self.get_joined_inputs(x,"longformer","sum", True,768))
        #print("Getting doc2vec selected embeddings")
        #self.split_data["doc2vec_selected_embedding"] = self.split_data["species_key"].apply(lambda x : self.get_selected_doc2vec_inputs(x))
        #print("Getting longformer selected embeddings")
        #self.split_data["longformer_selected_embedding"] = self.get_selected_longformer_inputs(self.split_data["species_key"])
        self.split_data.to_json(self.file_name, orient="records")
        print("Saved file, congrats")

if __name__ == "__main__":
    #"/scratch/izar/santacro/final_data/"
    root = "/scratch/izar/santacro/final_data/"
    cfg = parse_args()
    sp = SplitProcessor()
    sp.load_file(path=root, random_state = cfg.RANDOM_STATE, level=cfg.LEVEL, splitting=cfg.SPLITTING, split=cfg.SPLIT)
    sp.get_all_embeddings()