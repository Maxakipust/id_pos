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

emmisionCounts = {}
for tag in tags:
    emmisionCounts[tag] = {'UNK': 0}

transitionTotals = {}
for tag in tags:
    transitionTotals[tag] = 0

emmisionTotals = {'UNK': 0}

#init the count dicts for specific contexts
context_transition_counts = {}
for context in contexts:
    context_transition_counts[context] = {}
    for taga in tags:
        for tagb in tags:
            context_transition_counts[context][(taga, tagb)] = 0

context_emission_counts = {}
for context in contexts:
    context_emission_counts[context] = {}
    for tag in tags:
        context_emission_counts[context][tag] = {'UNK':0}

context_transition_totals = {}
for context in contexts:
    context_transition_totals[context] = {}
    for tag in tags:
        context_transition_totals[context][tag] = 0

context_emmision_totals = {}
for context in contexts:
    context_emmision_totals[context] = {'UNK': 0}

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
            identiferArr = ('SOI ' +row['IDENTIFIER'].lower() + ' EOI').split()
            #extract the grammar from the dataset
            posArr = ('SOI ' + row['GRAMMAR_PATTERN'] + ' EOI').split()
            #extract the context from the dataset
            context = contexts[int(row['CONTEXT'])-1]
            for i in range(0, len(identiferArr)-1):
                #clean up the word
                id = cleanUpWord(identiferArr[i])
                
                pos = posArr[i]
                nextpos = posArr[i+1]

                #record globals
                transitionCounts[(pos, nextpos)] = transitionCounts[(pos, nextpos)]+1
                transitionTotals[pos] = transitionTotals[pos] + 1

                emmisionCounts[pos][id] = emmisionCounts[pos][id] + 1 if id in emmisionCounts[pos] else 1
                emmisionTotals[id] = emmisionTotals[id] + 1 if id in emmisionTotals else 1

                #record context
                context_transition_counts[context][(pos, nextpos)] = context_transition_counts[context][(pos, nextpos)]+1
                context_transition_totals[context][pos] = context_transition_totals[context][pos] + 1

                context_emission_counts[context][pos][id] = context_emission_counts[context][pos][id] + 1 if id in context_emission_counts[context][pos] else 1
                context_emmision_totals[context][id] = context_emmision_totals[context][id] + 1 if id in context_emmision_totals[context] else 1



#replace any word with a global total count less then unkThreshold with UNK token in each of the other counts and totals
toDel = []
emmisionTotalsUnkCount = 0
for word in emmisionTotals:
    if emmisionTotals[word] <= unkThreshold:
        emmisionTotalsUnkCount = emmisionTotalsUnkCount + emmisionTotals[word]
        toDel.append(word)
        for tag in tags:
            if word in emmisionCounts[tag]:
                if 'UNK' not in emmisionCounts[tag]:
                    emmisionCounts[tag]['UNK'] = 0
                emmisionCounts[tag]['UNK'] = emmisionCounts[tag]['UNK'] + emmisionCounts[tag][word]
                del emmisionCounts[tag][word]
        for context in contexts:
            if word in context_emmision_totals[context]:
                if 'UNK' not in context_emmision_totals[context]:
                    context_emmision_totals[context]['UNK'] = 0
                context_emmision_totals[context]['UNK'] = context_emmision_totals[context]['UNK'] + context_emmision_totals[context][word]
                del context_emmision_totals[context][word]
            for tag in tags:
                if word in context_emission_counts[context][tag]:
                    if 'UNK' not in context_emission_counts[context][tag]:
                        context_emission_counts[context][tag]['UNK'] = 0
                    context_emission_counts[context][tag]['UNK'] = context_emission_counts[context][tag]['UNK'] + context_emission_counts[context][tag][word]
                    del context_emission_counts[context][tag][word]
# print("deleting", toDel)
for word in toDel:
    del emmisionTotals[word]
emmisionTotals['UNK'] = emmisionTotalsUnkCount

#calculate probablilites
transitionProbs = {}
for taga in tags:
    for tagb in tags:
        try:
            transitionProbs[(taga, tagb)] = transitionCounts[(taga, tagb)] / transitionTotals[taga]
        except ZeroDivisionError:
            transitionProbs[(taga, tagb)] = defaultProb

emmisionProbs = {}
for tag in tags:
    emmisionProbs[tag] = {}
    for id in emmisionCounts[tag]:
        try:
            emmisionProbs[tag][id] =  emmisionCounts[tag][id] / emmisionTotals[id] 
        except ZeroDivisionError:
            transitionProbs[tag][id] = defaultProb

context_transition_probs = {}
for context in contexts:
    context_transition_probs[context] = {}
    for taga in tags:
        for tagb in tags:
            try:
                context_transition_probs[context][(taga, tagb)] =  context_transition_counts[context][(taga,tagb)]/context_transition_totals[context][taga] 
            except ZeroDivisionError:
                context_transition_probs[context][(taga,tagb)] = defaultProb

context_emmision_probs = {}
for context in contexts:
    context_emmision_probs[context] = {}
    for tag in tags:
        context_emmision_probs[context][tag] = {}
        for id in context_emission_counts[context][tag]:
            try:
                context_emmision_probs[context][tag][id] = context_emission_counts[context][tag][id] / context_emmision_totals[context][id]
            except ZeroDivisionError:
                context_emmision_probs[context][tag][id] = defaultProb



# Viterbi
def run_viterbi(insent, context):

    ## Fill in some value for every state. But sometimes the emission probability will be 0 (e.g., 
    ## the probability that you see "the" when the tag is NN). Account for this possibility *and* for the 
    ## possibility of an OOV, by assigning defaultprob if the emission probability is 0.
    
    defaultprob =  0.00000000001

    ## Create the trellis and backpointer itself as a 2D np array
    stateprobs = np.zeros(shape=(len(insent), len(tags)))
    backpointers = np.zeros(shape=(len(insent), len(tags)))

    ## Initialize state 0, which must be the beginning of sentence (BOS) tag with probability 1
    stateprobs[0,tags.index("SOI")] = 1.0

    ## Fill in the rest of the trellis, starting with state 1
    ## Populate both state probabilities and backpointers

    for i in range(1, len(insent)):
        #for each pos
        for (posIdx, pos) in enumerate(tags):
            #calc argmax
            args = [ () for i in range(len(tags))]
            #for each prev pos
            for (prevIdx, prevPos) in enumerate(tags):
                #get the prev final pßrob
                prevProb = stateprobs[i-1, prevIdx]
                #get the transition prob
                contexttprob = context_transition_probs[context][prevPos, pos] if (prevPos, pos) in context_transition_probs[context] else defaultprob
                globaltprob = transitionProbs[prevPos, pos] if (prevPos, pos) in transitionProbs else defaultprob
                tprob = (contexttprob + globaltprob) / 2
                #calculate the prev prob * transition and save it with the pos
                args[prevIdx] = (prevProb * tprob, prevPos)
                
            #get the max pos and value from prev
            argmax = max(args, key= lambda x: x[0])
            #get emission probablity for the context
            contextWord = insent[i] #if insent[i] in context_emmision_probs[context][pos] else 'UNK'
            if contextWord in context_emmision_probs[context][pos] and context_emmision_probs[context][pos][contextWord] != 0:
                contextEmitProb = context_emmision_probs[context][pos][contextWord]
            else:
                contextEmitProb = defaultProb
            #get emission probablility globally
            globalWord = insent[i] #if insent[i] in context_emmision_probs[context][pos] else 'UNK'
            if globalWord in emmisionProbs[pos] and emmisionProbs[pos][globalWord] != 0:
                globalEmitProb = emmisionProbs[pos][globalWord]
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
    maxprob = stateprobs[len(insent)-1,tags.index("EOI")]
    maxprobid = backpointers[len(insent)-1,tags.index("EOI")]

    for i in range(len(insent)-2, 0, -1):
        # add the maxprob pos to posseq
        posseq = tags[maxprobid.astype(int)]+ " " + posseq
        # calculate the next maxprobid
        maxprobid = backpointers[i, maxprobid.astype(int)]
    posseq = "BOI " + posseq
    
    return posseq

success = 0
fail = 0

def run_ensemble(type, name, context):
    name = "%20".join(name.split())
    response = requests.get(f"http://localhost:5000/{type}/{name}/{context}")
    response = response.text.split(',')
    result = []
    for section in response:
        result.append(section.split('|')[1])
    return " ".join(result)

#run tests
with open('data/orig_unseen_testing_data.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    prevId = ""
    for row in reader:
        if row['IDENTIFIER'] != prevId:
            idArr = ('SOI ' +row['IDENTIFIER'].lower() + ' EOI').split()
            idArr = list(map(cleanUpWord, idArr))
            context = contexts[int(row['CONTEXT']) - 1]
            calcPOS = run_viterbi(idArr, context).split()
            actualPOS = ('BOI '+ row['GRAMMAR_PATTERN'] + ' EOI').split()
            for (index, actual) in enumerate(actualPOS):
                calc = calcPOS[index]
                if calc == actual:
                    success = success + 1
                else:
                    fail = fail + 1
                    # print("word:", idArr[index])
                    # print("expected:", actual)
                    # print("calc:", calc)
                    # print()
            
            # ensemble_result = run_ensemble(row["TYPE"], row["IDENTIFIER"], contexts[int(row['CONTEXT'])-1])
            # actualPOS = actualPOS[1:-1]
            # for (index, actual) in enumerate(actualPOS):
            #     calc = ensemble_result[index]
            #     if calc == actual:
            #         ensemble_success = ensemble_success + 1
            #     else:
            #         ensemble_fail = ensemble_fail + 1
                    

print("success",success)
print("fail", fail)
print("acc",success/(success+fail))

# print("nltk success",ensemble_success)
# print("nltk fail", ensemble_fail)
# print("nltk acc",ensemble_success/(ensemble_success+ensemble_fail))
