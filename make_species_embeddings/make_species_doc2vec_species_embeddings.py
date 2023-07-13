import json
from tqdm import tqdm, trange
import os
from gensim.models.doc2vec import Doc2Vec
from gensim.corpora.wikicorpus import tokenize
from gensim.models.keyedvectors import KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import pickle5 as pickle

class WikiTextCleaner():
    """Remove templates and unwanted text from textual inputs"""
    def __init__(self, remove_stopwords=True):
        self.split_words = ["External links", "References", "See also"]
        self.forbidden_words = ["==", "\n"]
        self.remove_stopwords = remove_stopwords
        if self.remove_stopwords:
            self.tokenizer = word_tokenize
            self.stop_words = stopwords.words("english")

    def clean_text(self, text, cut_useless_paragraphs=True):
        ## Cut out references
        if cut_useless_paragraphs:
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

if __name__ == "__main__":
    root = "/scratch/izar/santacro/final_data/species/"
    with open('/scratch/izar/santacro/models/doc2vec_light.pickle', 'rb') as handle:
        doc2vec = pickle.load(handle)
    """doc2vec = Doc2Vec.load('/data/nicola/WSH/models/doc2vec_dbow.model')
    temp = doc2vec.dv.vector_size
    doc2vec.dv = None
    doc2vec.dv = KeyedVectors(vector_size=temp)"""
    model = lambda x : doc2vec.infer_vector(tokenize(x))
    cleaner = WikiTextCleaner(remove_stopwords=False)
    for file in tqdm(os.listdir(root)):
        with open(root + file, "r") as fp:
            file_content = json.load(fp)
        wiki_description = file_content["page_text"]
        wiki_description = cleaner.clean_text(wiki_description)
        embedded = model(wiki_description)
        #file_content["longformerLastHiddenState"] = embedded["last_hidden_state"][0].to(device="cpu").tolist()
        file_content["doc2vec_embedding"] = embedded.tolist()
        with open(root + file, "w") as fp:
            json.dump(file_content,fp)
        #with open(root + file, "r") as fp:
        #    file_content = json.load(fp)