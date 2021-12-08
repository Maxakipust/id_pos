import numpy as np
import json
import common
import csv

#generate the probability files from the training data

# with open('data/orig_training_data.csv', newline='') as csvfile:
def generate_probabilities(infile, outEmissionProbs, outTransitionProbs, outContextEmissionProbs, outContextTransitionProbs):
    infile.seek(0)
    outEmissionProbs.seek(0)
    outTransitionProbs.seek(0)
    outContextEmissionProbs.seek(0)
    outContextTransitionProbs.seek(0)
    #init the count dicts for global counts
    transitionCounts = {}
    for taga in common.tags:
        for tagb in common.tags:
            transitionCounts[(taga, tagb)] = 0

    emissionCounts = {}
    for tag in common.tags:
        emissionCounts[tag] = {}

    transitionTotals = {}
    for tag in common.tags:
        transitionTotals[tag] = 0

    emissionTotals = {}

    #init the count dicts for specific contexts
    contextTransitionCounts = {}
    for context in common.contexts:
        contextTransitionCounts[context] = {}
        for taga in common.tags:
            for tagb in common.tags:
                contextTransitionCounts[context][(taga, tagb)] = 0

    contextEmissionCounts = {}
    for context in common.contexts:
        contextEmissionCounts[context] = {}
        for tag in common.tags:
            contextEmissionCounts[context][tag] = {}

    contextTransitionTotals = {}
    for context in common.contexts:
        contextTransitionTotals[context] = {}
        for tag in common.tags:
            contextTransitionTotals[context][tag] = 0

    contexEmissionTotals = {}
    for context in common.contexts:
        contexEmissionTotals[context] = {}
    
    reader = csv.DictReader(infile)
    prevId = ""
    for row in reader:
        #since the dataset contains an entry for each word, we only want to run this for each identifer
        if row['IDENTIFIER'] != prevId:
            prevId = row['IDENTIFIER']
            #get rid of case and add start and end words
            identifierArr = ('SOI ' +row['IDENTIFIER'].lower() + ' EOI').split()
            #extract the grammar from the dataset
            posArr = ('SOI ' + row['GRAMMAR_PATTERN'] + ' EOI').split()
            #extract the context from the dataset
            context = row['CONTEXT']
            try:
                context_int = int(context)
                context = common.contexts[context_int - 1]
            except:
                context = row['CONTEXT']
            for i in range(0, len(identifierArr)-1):
                #clean up the word
                id = common.cleanUpWord(identifierArr[i])
                
                pos = posArr[i]
                nextpos = posArr[i+1]

                #record globals
                transitionCounts[(pos, nextpos)] = transitionCounts[(pos, nextpos)]+1
                transitionTotals[pos] += + 1

                emissionCounts[pos][id] = emissionCounts[pos][id] + 1 if id in emissionCounts[pos] else 1
                emissionTotals[id] = emissionTotals[id] + 1 if id in emissionTotals else 1

                #record context
                contextTransitionCounts[context][(pos, nextpos)] = contextTransitionCounts[context][(pos, nextpos)]+1
                contextTransitionTotals[context][pos] += 1

                contextEmissionCounts[context][pos][id] = contextEmissionCounts[context][pos][id] + 1 if id in contextEmissionCounts[context][pos] else 1
                contexEmissionTotals[context][id] = contexEmissionTotals[context][id] + 1 if id in contexEmissionTotals[context] else 1

    #calculate probablilites
    transitionProbs = {}
    for taga in common.tags:
        for tagb in common.tags:
            try:
                transitionProbs[(taga, tagb)] = transitionCounts[(taga, tagb)] / transitionTotals[taga]
            except ZeroDivisionError:
                transitionProbs[(taga, tagb)] = common.defaultProb

    emissionProbs = {}
    for tag in common.tags:
        emissionProbs[tag] = {}
        for id in emissionCounts[tag]:
            try:
                emissionProbs[tag][id] =  emissionCounts[tag][id] / emissionTotals[id] 
            except ZeroDivisionError:
                emissionProbs[tag][id] = common.defaultProb

    contextTransitionProbs = {}
    for context in common.contexts:
        contextTransitionProbs[context] = {}
        for taga in common.tags:
            for tagb in common.tags:
                try:
                    contextTransitionProbs[context][(taga, tagb)] =  contextTransitionCounts[context][(taga,tagb)]/contextTransitionTotals[context][taga] 
                except ZeroDivisionError:
                    contextTransitionProbs[context][(taga,tagb)] = common.defaultProb

    contextemissionProbs = {}
    for context in common.contexts:
        contextemissionProbs[context] = {}
        for tag in common.tags:
            contextemissionProbs[context][tag] = {}
            for id in contextEmissionCounts[context][tag]:
                try:
                    contextemissionProbs[context][tag][id] = contextEmissionCounts[context][tag][id] / contexEmissionTotals[context][id]
                except ZeroDivisionError:
                    contextemissionProbs[context][tag][id] = common.defaultProb

    #output files
    # with open('model/emissionProbs.txt', 'w+') as outfile:
    for tag in emissionProbs:
        for id in emissionProbs[tag]:
            outEmissionProbs.write(f"{tag} {id}: {emissionProbs[tag][id]}\n")

    # with open('model/transitionProbs.txt', 'w+') as outfile:
    for taga in common.tags:
        for tagb in common.tags:
            outTransitionProbs.write(f"{taga} {tagb}: {transitionProbs[(taga,tagb)]}\n")

    # with open('model/contextTransitionProbs.txt', 'w+') as outfile:
    for context in common.contexts:
        for taga in common.tags:
            for tagb in common.tags:
                outContextTransitionProbs.write(f"{context} {taga} {tagb}: {contextTransitionProbs[context][(taga,tagb)]}\n")

    # with open('model/contextEmissionProbs.txt', 'w+') as outfile:
    for context in common.contexts:
        for tag in contextemissionProbs[context]:
            for id in contextemissionProbs[context][tag]:
                outContextEmissionProbs.write(f"{context} {tag} {id}: {contextemissionProbs[context][tag][id]}\n")