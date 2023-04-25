import os
import json
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import random

class WikiTextCleaner():
    """Remove templates and unwanted text from textual inputs"""
    def __init__(self, remove_stopwords=True):
        self.split_words = ["External links", "References", "See also"]
        self.forbidden_words = ["===", "==", "\n"]
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
        return paragraphs, name


if __name__ == "__main__":
    #df = pd.DataFrame(columns=["species_file","binomial_name","paragraph_index","paragraph_class","paragraphe_file"])
    #df.to_json("annotated_data.json", orient="records")
    species_path = "/data/nicola/WSH/final_data/species/"
    species_list = os.listdir(species_path)
    ## Species used in train, test and/or val sets
    used_species = pd.read_json("/data/nicola/WSH/final_data/L2_species_keys.json", orient="records")["species_key"].tolist()
    used_species = [str(spe)+".json" for spe in used_species]
    ## Species to be used for annotation
    unused_species = [spe for spe in species_list if spe not in used_species]
    annotated_data = pd.read_json("./annotated_data.json", orient="records")
    #annotated_data = pd.DataFrame(columns=["species_file","binomial_name","paragraph_index","paragraph_class","paragraphe_file"])
    ## Remove previously annotated files
    unused_species = [spe for spe in unused_species if spe not in annotated_data["species_file"].unique().tolist()]
    ## Shuffle species
    random.shuffle(unused_species)

    #unused_species = ["2359706.json"]

    pr = ParagraphsRetriever(root=species_path)
    ## Iterate over files
    for file in unused_species:
        ## Gather paragraphs
        paragraphs, name = pr.get_paragraphs(file)
        ## Set temporary df for the file contents
        file_df = pd.DataFrame(columns=["species_file","binomial_name","paragraph_index","paragraph_class","paragraphe_file"])
        print(f"=========\nFile {file} Species {name}, {len(paragraphs)} paragraphs\n=========\n")
        ## Iterate over pars
        for par in paragraphs:
            ## Annotate par
            a = input(par+ "\n")
            if a in ["y", "n"]:
                ## Log annotations
                id = paragraphs.index(par)
                par_file_name = f"{file[:-5]}_{id:03d}.txt"
                file_df.loc[len(file_df)] = [file, name, id, a, par_file_name]
                ## Save in file
                if a == "y":
                    with open(f"./positives/{par_file_name}", 'w') as f:
                        f.write(par)
                else:
                    with open(f"./negatives/{par_file_name}", 'w') as f:
                        f.write(par)
            else:
                raise ValueError("value has to be y of n (yes or no)")
        ## Update annotated data file
        annotated_data = pd.concat([annotated_data,file_df], ignore_index=True)
        annotated_data.to_json("annotated_data.json", orient="records")