import numpy as np
import requests

defaultProb = 0.000001
unkThreshold = 1

#an array with all the contexts
contexts = [
    'ATTRIBUTE',
    'CLASS',
    'DECLARATION',
    'FUNCTION',
    'PARAMETER'
]
#an array with all the tags
tags = [
    'SOI',
    'N',
    'DT',
    'CJ',
    'P',
    'NPL',
    'NM',
    'NM',
    'V',
    'VM',
    'PR',
    'D',
    'PRE',
    'EOI'
]

#init the count dicts for global counts
transitionCounts = {}
for taga in tags:
    for tagb in tags:
        transitionCounts[(taga, tagb)] = 0

emissionCounts = {}
for tag in tags:
    emissionCounts[tag] = {'UNK': 0}

transitionTotals = {}
for tag in tags:
    transitionTotals[tag] = 0

emissionTotals = {'UNK': 0}

#init the count dicts for specific contexts
contextTransitionCounts = {}
for context in contexts:
    contextTransitionCounts[context] = {}
    for taga in tags:
        for tagb in tags:
            contextTransitionCounts[context][(taga, tagb)] = 0

contextEmissionCounts = {}
for context in contexts:
    contextEmissionCounts[context] = {}
    for tag in tags:
        contextEmissionCounts[context][tag] = {'UNK':0}

contextTransitionTotals = {}
for context in contexts:
    contextTransitionTotals[context] = {}
    for tag in tags:
        contextTransitionTotals[context][tag] = 0

contexEmissionTotals = {}
for context in contexts:
    contexEmissionTotals[context] = {'UNK': 0}

#remove numbers from words
def cleanUpWord(id):
    try:
        idInt = int(id)
        return "NUM"
    except:
        return id

#calculate counts
import csv
with open('data/orig_training_data.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
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
            context = contexts[int(row['CONTEXT'])-1]
            for i in range(0, len(identifierArr)-1):
                #clean up the word
                id = cleanUpWord(identifierArr[i])
                
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



#replace any word with a global total count less then unkThreshold with UNK token in each of the other counts and totals
toDel = []
emissionTotalsUnkCount = 0
for word in emissionTotals:
    if emissionTotals[word] <= unkThreshold:
        emissionTotalsUnkCount = emissionTotalsUnkCount + emissionTotals[word]
        toDel.append(word)
        for tag in tags:
            if word in emissionCounts[tag]:
                if 'UNK' not in emissionCounts[tag]:
                    emissionCounts[tag]['UNK'] = 0
                emissionCounts[tag]['UNK'] += emissionCounts[tag][word]
                del emissionCounts[tag][word]
        for context in contexts:
            if word in contexEmissionTotals[context]:
                if 'UNK' not in contexEmissionTotals[context]:
                    contexEmissionTotals[context]['UNK'] = 0
                contexEmissionTotals[context]['UNK'] += contexEmissionTotals[context][word]
                del contexEmissionTotals[context][word]
            for tag in tags:
                if word in contextEmissionCounts[context][tag]:
                    if 'UNK' not in contextEmissionCounts[context][tag]:
                        contextEmissionCounts[context][tag]['UNK'] = 0
                    contextEmissionCounts[context][tag]['UNK'] += contextEmissionCounts[context][tag][word]
                    del contextEmissionCounts[context][tag][word]
# print("deleting", toDel)
for word in toDel:
    del emissionTotals[word]
emissionTotals['UNK'] = emissionTotalsUnkCount

#calculate probablilites
transitionProbs = {}
for taga in tags:
    for tagb in tags:
        try:
            transitionProbs[(taga, tagb)] = transitionCounts[(taga, tagb)] / transitionTotals[taga]
        except ZeroDivisionError:
            transitionProbs[(taga, tagb)] = defaultProb

emissionProbs = {}
for tag in tags:
    emissionProbs[tag] = {}
    for id in emissionCounts[tag]:
        try:
            emissionProbs[tag][id] =  emissionCounts[tag][id] / emissionTotals[id] 
        except ZeroDivisionError:
            transitionProbs[tag][id] = defaultProb

contextTransitionProbs = {}
for context in contexts:
    contextTransitionProbs[context] = {}
    for taga in tags:
        for tagb in tags:
            try:
                contextTransitionProbs[context][(taga, tagb)] =  contextTransitionCounts[context][(taga,tagb)]/contextTransitionTotals[context][taga] 
            except ZeroDivisionError:
                contextTransitionProbs[context][(taga,tagb)] = defaultProb

contextemissionProbs = {}
for context in contexts:
    contextemissionProbs[context] = {}
    for tag in tags:
        contextemissionProbs[context][tag] = {}
        for id in contextEmissionCounts[context][tag]:
            try:
                contextemissionProbs[context][tag][id] = contextEmissionCounts[context][tag][id] / contexEmissionTotals[context][id]
            except ZeroDivisionError:
                contextemissionProbs[context][tag][id] = defaultProb



# Viterbi
def runViterbi(identifier, context):
    identifier.insert(0,'BOI')
    identifier.append('EOI')

    ## Fill in some value for every state. But sometimes the emission probability will be 0 (e.g., 
    ## the probability that you see "the" when the tag is NN). Account for this possibility *and* for the 
    ## possibility of an OOV, by assigning defaultprob if the emission probability is 0.
    
    defaultprob =  0.00000000001

    ## Create the trellis and backpointer itself as a 2D np array
    stateprobs = np.zeros(shape=(len(identifier), len(tags)))
    backpointers = np.zeros(shape=(len(identifier), len(tags)))

    ## Initialize state 0, which must be the beginning of sentence (BOS) tag with probability 1
    stateprobs[0,tags.index("SOI")] = 1.0

    ## Fill in the rest of the trellis, starting with state 1
    ## Populate both state probabilities and backpointers

    for i in range(1, len(identifier)):
        #for each pos
        for (posIdx, pos) in enumerate(tags):
            #calc argmax
            args = [ () for i in range(len(tags))]
            #for each prev pos
            for (prevIdx, prevPos) in enumerate(tags):
                #get the prev final pßrob
                prevProb = stateprobs[i-1, prevIdx]
                #get the transition prob
                contexttprob = contextTransitionProbs[context][prevPos, pos] if (prevPos, pos) in contextTransitionProbs[context] else defaultprob
                globaltprob = transitionProbs[prevPos, pos] if (prevPos, pos) in transitionProbs else defaultprob
                tprob = (contexttprob + globaltprob) / 2
                #calculate the prev prob * transition and save it with the pos
                args[prevIdx] = (prevProb * tprob, prevPos)
                
            #get the max pos and value from prev
            argmax = max(args, key= lambda x: x[0])
            #get emission probablity for the context
            contextWord = identifier[i] #if insent[i] in contextemissionProbs[context][pos] else 'UNK'
            if contextWord in contextemissionProbs[context][pos] and contextemissionProbs[context][pos][contextWord] != 0:
                contextEmitProb = contextemissionProbs[context][pos][contextWord]
            else:
                contextEmitProb = defaultProb
            #get emission probablility globally
            globalWord = identifier[i] #if insent[i] in contextemissionProbs[context][pos] else 'UNK'
            if globalWord in emissionProbs[pos] and emissionProbs[pos][globalWord] != 0:
                globalEmitProb = emissionProbs[pos][globalWord]
            else:
                globalEmitProb = defaultProb
            #calculate the final emit prob as the average of the context prob and the global prob
            emitProb = (contextEmitProb + globalEmitProb)/2
            #calculate the final probablility
            finalprob = argmax[0] * emitProb
            #record results
            backpointers[i, posIdx] = tags.index(argmax[1])
            stateprobs[i,posIdx] = finalprob



    ## Store the string of POS tags in a variable called posseq
    posseq = "EOI"

    ## After populating the full trellis and backpointer trellis, print out the proposed POS tag sequence by traversing
    ## the backpointers from the backpointer in the last column that corresponds to "EOS" (end of sentence tag).

    ## Getting started
    #start at the EOI tag
    maxprob = stateprobs[len(identifier)-1,tags.index("EOI")]
    maxprobid = backpointers[len(identifier)-1,tags.index("EOI")]

    for i in range(len(identifier)-2, 0, -1):
        # add the maxprob pos to posseq
        posseq = tags[maxprobid.astype(int)]+ " " + posseq
        # calculate the next maxprobid
        maxprobid = backpointers[i, maxprobid.astype(int)]
    posseq = "BOI " + posseq
    
    return posseq

# ensemble_success = 0
# ensemble_fail = 0

success = 0
fail = 0

# def run_ensemble(type, name, context):
#     try:
#         name = "_".join(name)
#         if context == "FUNCTION":
#             name += "()"
#         response = requests.get(f"http://localhost:5000/{type}/{name}/{context}")
#         print("response", response.text)
#         response = response.text.split(',')
#         result = []
#         for section in response:
#             result.append(section.split('|')[1])
#         return " ".join(result)
#     except:
#         return ""

#run tests
with open('data/orig_unseen_testing_data.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    prevId = ""
    for row in reader:
        if row['IDENTIFIER'] != prevId:
            prevId = row['IDENTIFIER']
            # idArr = ('SOI ' +row['IDENTIFIER'].lower() + ' EOI').split()
            idArr = row['IDENTIFIER'].lower().split()
            idArr = list(map(cleanUpWord, idArr))
            print(idArr)
            context = contexts[int(row['CONTEXT']) - 1]
            # calcPOS = runViterbi(idArr, context).split()
            # ensemblePOS = run_ensemble("int", idArr, context).split()
            # print("ensemble", ensemblePOS)
            calcPOS = runViterbi(idArr, context).split()[1:-1]
            print("calc", calcPOS)
            # actualPOS = ('BOI '+ row['GRAMMAR_PATTERN'] + ' EOI').split()
            actualPOS = row['GRAMMAR_PATTERN'].split()
            print("actual", actualPOS)
            for (index, actual) in enumerate(actualPOS):
                # try:
                #     calc = ensemblePOS[index]
                #     if calc == actual:
                #         ensemble_success += 1
                #         # print("ensemble_success")
                #     else:
                #         ensemble_fail += 1
                #         print("ensemble_fail", actual, calc)
                # except:
                #     ensemble_fail += 1
                calc = calcPOS[index]
                if calc == actual:
                    success += 1
                    # print("success")
                else:
                    fail += 1
                    print("fail", actual, calc)
                
                    # print("word:", idArr[index])
                    # print("expected:", actual)
                    # print("calc:", calc)
                    # print()
            print()
                    

print("success",success)
print("fail", fail)
print("acc",success/(success+fail))

# print("ensemble_success",ensemble_success)
# print("ensemble_fail", ensemble_fail)
# print("ensemble_acc",ensemble_success/(ensemble_success+ensemble_fail))
