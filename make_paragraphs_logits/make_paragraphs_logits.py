from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import pandas as pd
import json
from tqdm import tqdm, trange
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
CUDA_LAUNCH_BLOCKING=1


class WikiTextCleaner():
    """Remove templates and unwanted text from textual inputs"""
    def __init__(self, remove_stopwords=True):
        self.split_words = ["External links", "References", "See also"]
        self.forbidden_words = ["==", "\n"]
        self.remove_stopwords = remove_stopwords
        if self.remove_stopwords:
            self.tokenizer = word_tokenize
            self.stop_words = stopwords.words("english")

    def clean_text(self, text):
        ## Cut out references
        for word in self.split_words:
            text = text.split(word)[0]
        ## Remove remaining templates
        text = re.sub('<[^>]+>', '', text)
        ## Remove stopwords
        if self.remove_stopwords:
            word_tokens = self.tokenizer(text)
            filtered = [w for w in word_tokens if w.lower() not in self.stop_words]
            text = " ".join(filtered)
        ## Remove unwanted words
        for word in self.forbidden_words:
            text = text.replace(word, "")
        ## Remove multiple spaces
        text = re.sub('\ \ +', ' ', text)
        return text
    
class ParagraphsRetriever():
    """Get paragraphs from file"""
    def __init__(self, root):
        self.root = root
        self.cleaner = WikiTextCleaner(remove_stopwords=False)
    
    def get_paragraphs(self,file):
        ## Read file
        with open(self.root+file,"r") as fp:
            content = json.load(fp)
        ## Parse paragraphs
        name = content["binomial_name"]
        paragraphs = content["page_text"].split("\n \n")
        ## Clean unwanted expressions
        for i in range(len(paragraphs)):
            paragraphs[i] = self.cleaner.clean_text(paragraphs[i])
        ## Remove small paragraphs
        paragraphs = [p for p in paragraphs if len(p)>20]
        return paragraphs, name, content

if __name__ == "__main__":
    root = "./../final_data/species/"
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    preprocessing = lambda x : tokenizer(x, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english").to(device="cuda:0")
    model.load_state_dict(torch.load("./../models/bestmodel_distilbert.pth"))
    model.eval()
    pr = ParagraphsRetriever(root=root)

    species_path = "./../final_data/species/"
    species_list = os.listdir(species_path)
    """used_species = pd.read_json("/scratch/izar/santacro/final_data/1_L2_species_keys.json", orient="records")["species_key"].tolist()
    used_species = [str(spe)+".json" for spe in used_species]"""

    for file in tqdm(species_list):
        paragraphs, _, content = pr.get_paragraphs(file)

        scores_positive = []
        for par in paragraphs:
            input = preprocessing(par)
            logits = model(input.input_ids.to(dtype=torch.int64, device="cuda:0"), input.attention_mask.to(dtype=torch.int64, device="cuda:0")).logits
            score_positive = float(torch.nn.functional.sigmoid(logits).squeeze().cpu().detach()[1])
            scores_positive.append(score_positive)
        content["paragraphs"] = paragraphs
        content["paragraphs_eco_scores"] = scores_positive
        
        with open(root + file, "w") as fp:
            json.dump(content,fp)

        """with open(root + file, "r") as fp:
            file_content = json.load(fp)
        for elem in file_content:
            print(elem)
        print(file_content["binomial_name"])
        print(file_content["page_title"])
        print(file_content["paragraphs_eco_scores"])
        break"""