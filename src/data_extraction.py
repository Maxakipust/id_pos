import csv
import common
import re
import random
import augment_labeled_data

globalexpr = re.compile('(.*) (.*): (.*)')


def pos_tag_with_max(in_emission_probs):
    emissionProbs = {}
    in_emission_probs.seek(0)
    for line in in_emission_probs.readlines():
        result = globalexpr.match(line)
        tag = result.group(1)
        id = result.group(2)
        prob = float(result.group(3))
        if tag not in emissionProbs:
            emissionProbs[tag] = {}
        emissionProbs[tag][id] = prob
    
    def run(id,ctx):
        tags = []
        for word in id:
            probs = []
            for tag in common.tags:
                if tag in emissionProbs:
                    if word in emissionProbs[tag]:
                        probs.append((tag, emissionProbs[tag][word]))
                    else:
                        probs.append((tag, 0))
            m = max(probs, key= lambda x: x[1])
            if m[1] == 0:
                tags.append("UNK")
            else:
                tags.append(m[0])
        return tags
    return run

def get_data():
    # get the counts of each pos tag and context in the training data 

    pos_counts = {}
    context_counts = {}

    with open("data/orig_unseen_testing_data.csv", "r+") as infile:
        reader = csv.DictReader(infile)
        prevId = ""
        for row in reader:
            if not row["IDENTIFIER"] == prevId:
                prevId = row["IDENTIFIER"]
                pattern = row['GRAMMAR_PATTERN'].split()
                for p in pattern:
                    if p not in pos_counts:
                        pos_counts[p] = 0
                    pos_counts[p] += 1

                context = row['CONTEXT']
                try:
                    context_int = int(context)
                    context = common.contexts[context_int - 1]
                except:
                    context = row['CONTEXT']
                if context not in context_counts:
                    context_counts[context] = 0
                context_counts[context] += 1

    for x in pos_counts:
        print(x, pos_counts[x])
    # print(pos_counts)

    for x in context_counts:
        print(x, context_counts[x])
    # print(context_counts)

def extract_data():
    data = []

    with open("data/orig_training_data.csv", "r+") as infile:
        reader = csv.DictReader(infile)
        prevId = ""
        for row in reader:
            if not row["IDENTIFIER"] == prevId:
                prevId = row["IDENTIFIER"]
                data.append((row["IDENTIFIER"].split(" "), row["GRAMMAR_PATTERN"].split(" "), row["CONTEXT"]))
    
    random.shuffle(data)

    index = int(len(data) * 0.15)
    validation_data = data[:index]
    data = data[index:]

    with open("data/training_data.csv", "w+") as training_data:
        augment_labeled_data.write_data(data, training_data)
    
    with open("data/validation.csv", "w+") as validation:
        augment_labeled_data.write_data(validation_data, validation)
extract_data()