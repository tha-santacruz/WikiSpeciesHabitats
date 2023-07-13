from gensim.models.doc2vec import Doc2Vec
from gensim.models.keyedvectors import KeyedVectors
import pickle

if __name__ == "__main__":
    print("pruning doc2vec model to make it lighter")
    size = 768
    method = "dbow"
    dv_model = Doc2Vec.load(f'/data/nicola/WSH/models/doc2vec_{method}_{size}.model')
    print("loaded doc2vec")
    temp = dv_model.dv.vector_size
    dv_model.dv = None
    dv_model.dv = KeyedVectors(vector_size=temp)
    print("reset keyed vectors")
    with open(f'/data/nicola/WSH/models/doc2vec_{method}_light.pickle', 'wb') as handle:
        pickle.dump(dv_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("saved lighter model")