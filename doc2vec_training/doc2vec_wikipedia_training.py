import logging
import multiprocessing
from pprint import pprint

import smart_open
from gensim.corpora.wikicorpus import WikiCorpus, tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class TaggedWikiCorpus:
    def __init__(self, wiki_text_path):
        self.wiki_text_path = wiki_text_path
        
    def __iter__(self):
        for line in smart_open.open(self.wiki_text_path, encoding='utf8'):
            title, words = line.split('\t')
            yield TaggedDocument(words=words.split(), tags=[title])

if __name__ == "__main__":
    ## Creating corpus
    """wiki = WikiCorpus(
        "/data/nicola/WSH/wikipedia_dump/enwiki-latest-pages-articles.xml.bz2",  # path to the file you downloaded above enwiki-20221101-pages-articles1.xml-p1p41242.bz2",#
        tokenizer_func=tokenize,  # simple regexp; plug in your own tokenizer here
        metadata=True,  # also return the article titles and ids when parsing
        dictionary={},  # don't start processing the data yet
        )

    with smart_open.open("/data/nicola/WSH/wikipedia_dump/wiki.txt.gz", "w", encoding='utf8') as fout:
        for article_no, (content, (page_id, title)) in enumerate(wiki.get_texts()):
            title = ' '.join(title.split())
            if article_no % 500000 == 0:
                logging.info("processing article #%i: %r (%i tokens)", article_no, title, len(content))
            fout.write(f"{title}\t{' '.join(content)}\n")  # title_of_article [TAB] words of the article"""

    documents = TaggedWikiCorpus('/scratch/santacro/wiki.txt.gz')  # A streamed iterable; nothing in RAM yet.

    workers = 17  # multiprocessing.cpu_count() - 1  # leave one core for the OS & other stuff
    vector_size = 768
    epochs = 10

    ## PV-DBOW: paragraph vector in distributed bag of words mode
    """
    model = Doc2Vec(
        dm=0, dbow_words=1,  # dbow_words=1 to train word vectors at the same time too, not only DBOW
        vector_size=vector_size, window=8, epochs=epochs, workers=workers, max_final_vocab=1000000,
    )
    model.build_vocab(documents, progress_per=500000)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs, report_delay=30*60)
    model.save(f'/scratch/santacro/doc2vec_dbow_{vector_size}.model')
    """
    ## PV-DM: paragraph vector in distributed memory mode
    model = Doc2Vec(
        dm=1, dm_mean=1,  # use average of context word vectors to train DM
        vector_size=vector_size, window=8, epochs=epochs, workers=workers, max_final_vocab=1000000,
    )
    model.build_vocab(documents, progress_per=500000)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs, report_delay=30*60)
    model.save(f'/scratch/santacro/doc2vec_dm_{vector_size}.model')
