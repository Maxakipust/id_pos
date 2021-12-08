import gensim
from gensim.models import Word2Vec

# uses a word2vec model to establish relationships between words
# then use those relationships to extend the probabilities of known words to unknown words


#cuttoff for cos similarity between 2 words to consider them associated
cos_cutoff = 0.9

def word2vec_clustering(in_ids, in_probs, out_word2vec, out_probs):
    in_ids.seek(0)
    in_probs.seek(0)
    # out_word2vec.seek(0)
    out_probs.seek(0)

    #load in the unlabeled ids
    ids = in_ids.read().split("\n")
    data = []
    for id in ids:
        data.append(id.split())

    # Create word2vec model
    model = Word2Vec(sentences=data, vector_size=100, window=1, min_count=3, workers=16)
    model.save(out_word2vec)

    # load the existing probabilites
    probs = {}
    for line in in_probs:
        pos = line.split(': ')[0].split(' ')[0]
        word = line.split(': ')[0].split(' ')[1]
        prob = float(line.split(': ')[1])

        if word not in probs:
            probs[word] = {}
        probs[word][pos] = prob


    # extend the probabilites of words we dont know by extending the probabilites of the words that we do know as long as they are similar enough
    all_words = model.wv.key_to_index
    for word in all_words:
        if word in probs:
            for word2 in all_words:
                if word2 not in probs:
                    if model.wv.similarity(word, word2) > cos_cutoff:
                        probs[word2]=probs[word]

    #output new probs
    for word in probs:
        for pos in probs[word]:
            out_probs.write(f"{pos} {word}: {probs[word][pos]}\n")