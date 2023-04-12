import json
import os
import re
import pandas as pd

class WikiTextCleaner():
    """Remove templates and unwanted text from textual inputs"""
    def __init__(self):
        self.forbidden_words = ["==", "===", "\n"]

    def clean_text(self, text):
        ## Remove remaining templates
        text = re.sub('<[^>]+>', '', text)
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
        self.cleaner = WikiTextCleaner()
    
    def get_paragraphs(self,file):
        ## Read file
        with open(self.root+file,"r") as fp:
            content = json.load(fp)
        ## Parse paragraphs (and remove small ones)
        paragraphs = content["pageText"].split("\n \n")
        paragraphs = [p for p in paragraphs if len(p)>20]
        ## Clean unwanted expressions
        for i in range(len(paragraphs)):
            paragraphs[i] = self.cleaner.clean_text(paragraphs[i])
        return paragraphs

if __name__ == "__main__":
    root = "/data/nicola/WSH/WikiSpeciesHabitats/species/"
    pr = ParagraphsRetriever(root=root)

    for file in os.listdir(root):
        paragraphs = pr.get_paragraphs(file)
        for par in paragraphs:
            a = input(par+ "\n")
        
