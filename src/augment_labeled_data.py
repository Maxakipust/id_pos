import csv
import common
import random
from nltk.corpus import wordnet
import inflect

PERCENT_TO_SYN = 1.0
PERCENT_TO_INFLECT = 1.0

pos_2_wordnet = {
    "N": wordnet.NOUN,
    "NM":wordnet.NOUN,
    "V":wordnet.VERB,
    "VM":wordnet.VERB,
}

def load_vocab_and_data(infile):
    infile.seek(0)
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
            context = row['CONTEXT']
            try:
                context_int = int(context)
                context = common.contexts[context_int - 1]
            except:
                context = row['CONTEXT']
                
            data.append((identifierArr, posArr, context))
    return (vocab, data)


# with open('data/aug_training_data.csv', 'w+', newline='') as csvfile:
def write_data(data, outfile):
    outfile.seek(0)
    writer = csv.DictWriter(outfile, fieldnames=["IDENTIFIER", "GRAMMAR_PATTERN", "CONTEXT"])
    writer.writeheader()
    for d in data:
        (identifierArrOrig, posArr, context) = d
        writer.writerow({"IDENTIFIER":" ".join(identifierArrOrig), "GRAMMAR_PATTERN":" ".join(posArr), "CONTEXT": context})
p = inflect.engine()
# with open('data/orig_training_data.csv', newline='') as csvfile:
def syn_labeled_data(infile, outfile):
    infile.seek(0)
    outfile.seek(0)
    print("augmenting labeled data with synonyms")
    (vocab, data) = load_vocab_and_data(infile)
    print(f"original number of words: {len(vocab)}")
    print(f"original number of ids: {len(data)}")
    indexes_to_syn = random.sample(range(0, len(data)), int(PERCENT_TO_SYN*len(data)))
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
                        if posArr[word_index] == "N" or posArr[word_index] == "NM":
                            if (False == p.singular_noun(identifierArrOrig[word_index])) == (False == p.singular_noun(lemma.name())):
                                syns_have.append(lemma.name())
                                identifierArr = identifierArrOrig.copy()
                                identifierArr[word_index] = lemma.name()

                                data.append((identifierArr, posArr, context))
                                if lemma.name() not in vocab:
                                    vocab.append(lemma.name())
                        else:
                            syns_have.append(lemma.name())
                            identifierArr = identifierArrOrig.copy()
                            identifierArr[word_index] = lemma.name()

                            data.append((identifierArr, posArr, context))
                            if lemma.name() not in vocab:
                                vocab.append(lemma.name())
    print(f"final number of words: {len(vocab)}")
    print(f"final number of ids: {len(data)}")
    write_data(data, outfile)


def plural_labeled_data(infile, outfile):
    infile.seek(0)
    outfile.seek(0)
    print("augmenting labeled data with plural and singular")
    (vocab, data) = load_vocab_and_data(infile)
    print(f"original number of words: {len(vocab)}")
    print(f"original number of ids: {len(data)}")

    indexes_to_inflect = random.sample(range(0, len(data)), int(PERCENT_TO_INFLECT*len(data)))
    

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