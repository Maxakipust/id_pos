# Python program to generate word vectors using Word2Vec
  
import gensim
from gensim.models import Word2Vec

cos_cutoff = 0.9
  
#  Reads ‘alice.txt’ file
with open('data/unlabeled_ids.txt', newline='') as file:
    ids = file.read().split("\n")
    data = []
    for id in ids:
        data.append(id.split())
    
    print(f"creating model on {len(ids)} identifiers")
    # Create CBOW model
    model = Word2Vec(sentences=data, vector_size=100, window=1, min_count=1, workers=8)
    print("done making model")
    print("saving model as model/word2vec.model")
    model.save("model/word2vec.model")
    print("model saved")

print("loading emissionProbs")
probs = {}

with open('model/emissionProbs.txt', 'r') as infile:
    for line in infile:
        # pos, tag, prob
        
        pos = line.split(': ')[0].split(' ')[0]
        word = line.split(': ')[0].split(' ')[1]
        prob = float(line.split(': ')[1])

        if word not in probs:
            probs[word] = {}
        probs[word][pos] = prob

print("done loading emissionProbs")

all_words = model.wv.key_to_index
with open('model/emissionProbs.txt', 'a') as outfile:
    for word in all_words:
        if word in probs:
            for word2 in all_words:
                if word2 not in probs:
                    if model.wv.similarity(word, word2) > cos_cutoff:
                        for tag in probs[word]:
                            print(f"{word},{word2}")
                            outfile.write(f"{tag} {word2}: {probs[word][tag]}\n")
                        probs[word2]=probs[word]