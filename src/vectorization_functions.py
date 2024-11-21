from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import gensim.downloader as api
import numpy as np

# ---- Bag Of Words ----
def vectorize_bow(corpus):
    """
    Vectoriza un corpus utilizando Bag of Words.
    """
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

# ---- TF-IDF ----
def vectorize_tfidf(corpus):
    """
    Vectoriza un corpus utilizando TF-IDF.
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

# ---- Word2Vec ----
def vectorize_word2vec(corpus_tokens, w2v_model):
    """
    Vectoriza un corpus utilizando Word2Vec preentrenado.
    Cada documento se convierte en un vector promedio.
    """
    def get_avg_word2vec(tokens, model):
        valid_vectors = [model[word] for word in tokens if word in model]
        if valid_vectors:
            return sum(valid_vectors) / len(valid_vectors)
        else:
            return np.zeros(model.vector_size)

    vectors = [get_avg_word2vec(tokens, w2v_model) for tokens in corpus_tokens]
    
    return np.vstack(vectors)