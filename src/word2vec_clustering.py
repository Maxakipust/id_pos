import gensim
from gensim.models import Word2Vec

cos_cutoff = 0.9

def word2vec_clustering(in_ids, in_probs, out_word2vec, out_probs):
    in_ids.seek(0)
    in_probs.seek(0)
    # out_word2vec.seek(0)
    out_probs.seek(0)

    ids = in_ids.read().split("\n")
    data = []
    for id in ids:
        data.append(id.split())

    # print(f"creating model on {len(ids)} identifiers")
    # Create CBOW model
    model = Word2Vec(sentences=data, vector_size=100, window=1, min_count=3, workers=16)
    # print("done making model")
    # print("saving model as model/word2vec.model")
    model.save(out_word2vec)
    # print("model saved")

    # print("loading emissionProbs")
    probs = {}

    # with open('model/emissionProbs.txt', 'r') as infile:
    for line in in_probs:
        # pos, tag, prob
        
        pos = line.split(': ')[0].split(' ')[0]
        word = line.split(': ')[0].split(' ')[1]
        prob = float(line.split(': ')[1])

        if word not in probs:
            probs[word] = {}
        probs[word][pos] = prob

    # print("done loading emissionProbs")

    all_words = model.wv.key_to_index
    # with open('model/emissionProbs.txt', 'a') as outfile:
    for word in all_words:
        if word in probs:
            for word2 in all_words:
                if word2 not in probs:
                    if model.wv.similarity(word, word2) > cos_cutoff:
                        probs[word2]=probs[word]

    for word in probs:
        for pos in probs[word]:
            out_probs.write(f"{pos} {word}: {probs[word][pos]}\n")