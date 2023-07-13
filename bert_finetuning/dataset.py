import json
import os
import re
import tempfile
import time
import torch
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader

from transformers import DistilBertTokenizer

class BertTuningDataset():
    def __init__(self, path, split, preprocessing = None):
        """Class initialization"""
        ## Path and data
        self.path = path
        self.split = split
        self.preprocessing = preprocessing
        ## split
        df = pd.read_json(self.path + "annotated_data.json", orient="records")
        num_val = 20
        positives = df.loc[df["paragraph_class"]=="y"]
        positives_val = positives.sample(num_val, random_state=42)
        positives_val["split"] = "val"
        positives_train = positives.drop(positives_val.index)
        positives_train["split"] = "train"
        negatives = df.loc[df["paragraph_class"]=="n"]
        negatives_val = negatives.sample(num_val, random_state=42)
        negatives_val["split"] = "val"
        negatives_train = negatives.drop(negatives_val.index)
        negatives_train["split"] = "train"
        self.all_data = pd.concat([positives_train,positives_val,negatives_train,negatives_val])
        if self.split == "val":
            self.split_data =  self.all_data.loc[self.all_data["split"]=="val"].reset_index(drop=True)
        elif self.split == "train":
            all_train = self.all_data.loc[self.all_data["split"]=="train"]
            positives = all_train.loc[all_train["paragraph_class"]=="y"]
            negatives = all_train.loc[all_train["paragraph_class"]=="n"].sample(len(positives))
            self.split_data = pd.concat([negatives,positives]).reset_index(drop=True)
            #print(self.split_data.head())
        self.num_samples = len(self.split_data)
    
    def resample_negatives(self):
        """Resample negative examples to create balanced set"""
        all_train = self.all_data.loc[self.all_data["split"]=="train"]
        positives = all_train.loc[all_train["paragraph_class"]=="y"]
        negatives = all_train.loc[all_train["paragraph_class"]=="n"].sample(len(positives))
        self.split_data = pd.concat([negatives,positives]).reset_index(drop=True)
        #print(self.split_data.head())

    def __len__(self):
        """Mandatory lenght method"""
        return len(self.split_data)
    
    def __getitem__(self, idx):
        """Get input and target pairs"""
        ## Get sample, target
        entry = self.split_data.loc[idx]
        if entry["paragraph_class"] == "y":
            target = torch.tensor([0,1])
        elif entry["paragraph_class"] == "n":
            target = torch.tensor([1,0])
        ## Get input
        file_path = self.path + "all_examples/" + entry["paragraph_file"]
        with open(file_path,"r") as fp:
            input_text = fp.read()
        if self.preprocessing:
            input = self.preprocessing(input_text)
            return input.input_ids.squeeze(dim=0), input.attention_mask.squeeze(dim=0), target, input_text
        return input_text, target

if __name__ == "__main__":
    """Testing of the dataset class, for debugging"""

    root = "/data/nicola/WSH/bert_finetuning/"
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    prepr= lambda x : tokenizer(x, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    with tempfile.TemporaryDirectory() as temp_dir_path:
        print(temp_dir_path)

        ds = BertTuningDataset(path=root, split="train", preprocessing=prepr)
        #ds.split_data = ds.split_data[:4]
        dl = DataLoader(ds, shuffle=True, batch_size = 1)
        #batch = next(iter(dl))
        acc = torch.zeros(2)
        acc_texts = []
        for epoch in range(2):
            ds.resample_negatives()
            #ds.split_data = ds.split_data[:4]
            print("epoch")
            for batch in dl:
                #print(len(batch))
                #print(batch[0][0][:10])
                #print(batch[0][1])
                #print(batch[2][0])
                acc += batch[2][0]
                #print(batch[3][0])
                #print(batch[2])
                #print(batch[3][0])
                acc_texts.append(batch[3][0])
                #print(acc_texts)
        print(acc_texts)
        print(acc)
        print(len(list(set(acc_texts))))
            