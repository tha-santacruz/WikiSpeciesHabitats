from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from utils import ParagraphsRetriever
import pandas as pd
import json
from tqdm import tqdm, trange
import os
CUDA_LAUNCH_BLOCKING=1

if __name__ == "__main__":
    root = "/data/nicola/WSH/final_data/species/"
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    preprocessing = lambda x : tokenizer(x, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english").to(device="cuda:0")
    model.load_state_dict(torch.load("/data/nicola/WSH/checkpoints/bestmodel_distilbert.pth"))
    model.eval()
    pr = ParagraphsRetriever(root=root)

    species_path = "/data/nicola/WSH/final_data/species/"
    species_list = os.listdir(species_path)
    used_species = pd.read_json("/data/nicola/WSH/final_data/L2_species_keys.json", orient="records")["species_key"].tolist()
    used_species = [str(spe)+".json" for spe in used_species]

    for file in tqdm(used_species):
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