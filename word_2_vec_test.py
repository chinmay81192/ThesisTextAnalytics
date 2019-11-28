from gensim.models import Word2Vec


if __name__ == "__main__":
    model = Word2Vec.load("word2vec.model")
    print(model.most_similar('knowledge'))