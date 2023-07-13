import json
import os
import time

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class WikiSpeciesHabitats():
    def __init__(self, path, split, merge_method = "fusion", random_state = 1, target_base = "set", splitting="species", fraction = 0, embedding="agnostic", level="L1", fusion="sum", normalize="yes"):
        """Class initialization"""
        ## Configuration variables
        self.path = path
        self.split = split
        self.target_base = target_base
        self.splitting = splitting
        self.fraction = fraction
        self.random_state = random_state
        self.level = level
        self.merge_method = merge_method
        self.embedding = embedding
        self.fusion = fusion
        self.normalize = False
        if normalize == "yes":
            self.normalize = True
        ## Dataset loading
        if self.splitting == "progressive":
            if self.split == "test":
                self.split_data = pd.read_json(self.path + f"{self.random_state}_{self.level}_progressive_test_data.json", orient="records")
            else:
                self.split_data = pd.read_json(self.path + f"{self.random_state}_{self.level}_progressive_{int(self.fraction)}%_{self.split}_data.json", orient="records")
        else:
            self.split_data = pd.read_json(self.path + f"{self.random_state}_{self.level}_{self.splitting}_based_{self.split}_data.json", orient="records")
        self.species_keys = pd.read_json(self.path + f"{self.random_state}_{self.level}_species_keys.json", orient="records")
        self.habitats_keys = pd.read_json(self.path + f"{self.random_state}_{self.level}_habitats_keys.json", orient="records")
        ## Input sizes
        if self.embedding == "longformer":
            self.inputs_size = 768
        elif self.embedding == "doc2vec":
            self.inputs_size = 768
        elif self.embedding == "agnostic":
            self.inputs_size = len(self.species_keys)
        ## Inputs field
        if self.embedding == "agnostic":
            self.input_field = "agnostic_embedding"
        elif self.merge_method == "fusion":
            self.input_field = f"{self.embedding}_joined_embedding"
        elif self.merge_method == "selection":
            self.input_field = f"{self.embedding}_selected_embedding"
        ## Targets field
        self.target_field = f"{self.target_base}_based_class"
        ## Other useful info
        self.classes_codes = self.habitats_keys["class"]
        self.classes_names = self.habitats_keys["TypoCH_FR"].apply(lambda x : x[:20])
        self.num_classes = len(self.classes_codes)
        ## Positive weights
        all_targets = torch.tensor(self.split_data[self.target_field].values.tolist())
        self.positive_weight = (all_targets==0).sum(dim=0)/(all_targets==1).sum(dim=0)

    def get_inputs(self, species_keys):
        """Retrieve textual inputs from json files"""
        if self.embedding == "agnostic":
            input = torch.tensor(self.species_keys[self.species_keys["species_key"].isin(species_keys)].ID.values)
            input = F.one_hot(input, num_classes=self.inputs_size).sum(dim=0)
            return input
        else:
            merged_embeddings = torch.zeros((len(species_keys),self.inputs_size))
            i = 0
            ## Aggregate embeddings of present species using max pooling
            for key in species_keys:
                with open(self.path + f"species/{key}.json", "r") as fp:
                    file_content = json.load(fp)
                species_embeddings = file_content[self.embedding + "_embedding"]
                merged_embeddings[i,:] = torch.tensor(species_embeddings)
                i += 1
            if self.fusion == "max":
                merged_embeddings, _ = merged_embeddings.max(dim=0)
                if self.normalize:
                    merged_embeddings = merged_embeddings.div(merged_embeddings.norm())
            elif self.fusion == "prod":
                merged_embeddings = merged_embeddings.prod(dim=0)
                if self.normalize:
                    merged_embeddings = merged_embeddings.div(merged_embeddings.norm())
            elif self.fusion == "sum":
                merged_embeddings = merged_embeddings.sum(dim=0)
                if self.normalize:
                    merged_embeddings = merged_embeddings.div(merged_embeddings.norm())
            elif self.fusion == "mean":
                merged_embeddings = merged_embeddings.mean(dim=0)
                if self.normalize:
                    merged_embeddings = merged_embeddings.div(merged_embeddings.norm())
            else:
                raise Exception("invalid embeddings fusion method, must be 'max', 'prod', or 'mean'.")
            return merged_embeddings

    def __len__(self):
        """Mandatory lenght method"""
        return len(self.split_data)
    
    def __getitem__(self, idx):
        """Get input and target pairs"""
        entry = self.split_data.loc[idx]
        target = torch.tensor(entry[self.target_field])
        input = self.get_inputs(entry["species_key"])
        return input, target

if __name__ == "__main__":
    """Testing of the dataset class, for debugging"""

    root = "/data/nicola/WSH/final_data/"

    ds = WikiSpeciesHabitats(path=root, split="train", merge_method = "fusion", random_state = 5, target_base = "set", splitting="progressive", fraction = 40, embedding="doc2vec", level="L1")
    print(len(ds))
    ds.split_data = ds.split_data[:4]
    dl = DataLoader(ds, shuffle=False, batch_size = 2)
    batch = next(iter(dl))
    print(batch[0].size())
    print(batch[1].size())
    print(ds.positive_weight)

    #print(len(ds.species_keys))
"""
    start = time.time()
    for epoch in range(2):
        for i in dl:
            batch = i
            print("iterred")
            #print(len(batch))
            #print(batch[0][0].size())
            #print(batch[0][1])
            #print(batch[1])
            print(batch[0])
        print(f"time {time.time()-start}")
        start = time.time()"""
            