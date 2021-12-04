import csv
import common
import random
from nltk.corpus import wordnet

PERCENT_OF_UNK = 0.05
PERCENT_TO_SYN = 0.05

vocab = []
data = []

pos_2_wordnet = {
    "N": wordnet.NOUN,
    "NM":wordnet.NOUN,
    "V":wordnet.VERB,
    "VM":wordnet.VERB,
}

with open('data/orig_training_data.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    prevId = ""
    for row in reader:
        if row['IDENTIFIER'] != prevId:
            prevId = row['IDENTIFIER']
            identifierArr = (row['IDENTIFIER'].lower()).split()
            for word in identifierArr:
                if word not in vocab:
                    vocab.append(word)
            #extract the grammar from the dataset
            posArr = (row['GRAMMAR_PATTERN']).split()
            #extract the context from the dataset
            context = common.contexts[int(row['CONTEXT'])-1]
            data.append((identifierArr, posArr, context))
print(f"original number of words: {len(vocab)}")
print(f"original number of ids: {len(data)}")

# indexes_to_unk = random.sample(range(0, len(data)), int(PERCENT_OF_UNK*len(vocab)))
indexes_to_syn = random.sample(range(0, len(data)), int(PERCENT_TO_SYN*len(vocab)))
# for index in indexes_to_unk:
#     (identifierArrOrig, posArr, context) = data[index]
#     identifierArr = identifierArrOrig.copy()
#     word_to_unk = random.choice(range(0, len(identifierArr)))
#     identifierArr[word_to_unk] = "UNK"
#     data.append((identifierArr, posArr, context))

for index in indexes_to_syn:
    (identifierArrOrig, posArr, context) = data[index]
    
    word_indexes = [i for i in range(0, len(identifierArrOrig))]
    random.shuffle(word_indexes)
    for word_index in word_indexes:
        word_pos = posArr[word_index]
        if word_pos in pos_2_wordnet:
            wordnet_pos = pos_2_wordnet[word_pos]
        else:
            continue

        syns = wordnet.synsets(identifierArrOrig[word_index], pos=wordnet_pos)
        if len(syns) == 0:
            continue
        syns_have = [identifierArrOrig[word_index]]
        for syn in syns:
            lemmas = syn.lemmas()
            for lemma in lemmas:
                if lemma.name() not in syns_have:
                    # print(identifierArrOrig[word_index], lemma.name())
                    syns_have.append(lemma.name())
                    identifierArr = identifierArrOrig.copy()
                    identifierArr[word_index] = lemma.name()
                    data.append((identifierArr, posArr, context))
                    if lemma.name() not in vocab:
                        vocab.append(lemma.name())
        break
print(f"final number of words: {len(vocab)}")
print(f"final number of ids: {len(data)}")
with open('data/aug_training_data.csv', 'w+', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["IDENTIFIER", "GRAMMAR_PATTERN", "CONTEXT"])
    writer.writeheader()
    for d in data:
        (identifierArrOrig, posArr, context) = d
        writer.writerow({"IDENTIFIER":" ".join(identifierArrOrig), "GRAMMAR_PATTERN":" ".join(posArr), "CONTEXT": context})