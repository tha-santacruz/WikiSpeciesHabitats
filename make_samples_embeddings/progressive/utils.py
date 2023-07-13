from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

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