import csv
import common
import random
from nltk.corpus import wordnet
import inflect

PERCENT_OF_UNK = 0.05
PERCENT_TO_SYN = 0.05
PERCENT_TO_INFLECT = 0.5

pos_2_wordnet = {
    "N": wordnet.NOUN,
    "NM":wordnet.NOUN,
    "V":wordnet.VERB,
    "VM":wordnet.VERB,
}

def load_vocab_and_data(infile):
    vocab = []
    data = []
    reader = csv.DictReader(infile)
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
    return (vocab, data)


# with open('data/aug_training_data.csv', 'w+', newline='') as csvfile:
def write_data(data, outfile):
    writer = csv.DictWriter(outfile, fieldnames=["IDENTIFIER", "GRAMMAR_PATTERN", "CONTEXT"])
    writer.writeheader()
    for d in data:
        (identifierArrOrig, posArr, context) = d
        writer.writerow({"IDENTIFIER":" ".join(identifierArrOrig), "GRAMMAR_PATTERN":" ".join(posArr), "CONTEXT": context})

# with open('data/orig_training_data.csv', newline='') as csvfile:
def syn_labeled_data(infile, outfile):
    print("augmenting labeled data with synonyms")
    (vocab, data) = load_vocab_and_data(infile)
    print(f"original number of words: {len(vocab)}")
    print(f"original number of ids: {len(data)}")
    indexes_to_syn = random.sample(range(0, len(data)), int(PERCENT_TO_SYN*len(vocab)))
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
    write_data(data, outfile)


def plural_labeled_data(infile, outfile):
    print("augmenting labeled data with plural and singular")
    (vocab, data) = load_vocab_and_data(infile)
    print(f"original number of words: {len(vocab)}")
    print(f"original number of ids: {len(data)}")

    indexes_to_inflect = random.sample(range(0, len(data)), int(PERCENT_TO_INFLECT*len(vocab)))
    p = inflect.engine()

    for index in indexes_to_inflect:
        (identifierArrOrig, posArrOrig, context) = data[index]
        identifierArr = identifierArrOrig.copy()
        posArr = posArrOrig.copy()
        for (index, word) in enumerate(identifierArrOrig):
            if posArrOrig[index] == "N":
                plural = p.plural(word)
                identifierArr[index] = plural
                posArr[index] = "NPL"
            if posArrOrig[index] == "NPL":
                singular = p.singular_noun(word)
                identifierArr[index] = singular
                posArr[index] = "N"
        data.append((identifierArr, posArr, context))

    print(f"final number of words: {len(vocab)}")
    print(f"final number of ids: {len(data)}")
    write_data(data, outfile)