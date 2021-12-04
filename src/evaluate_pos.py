import numpy as np
import re
import common

globalexpr = re.compile('(.*) (.*): (.*)')
contextexpr = re.compile('(.*) (.*) (.*): (.*)')

def load_probs(in_emission_probs, in_transition_probs, in_context_emission_probs, in_context_transition_probs, global_transition_weight, global_emission_weight, context_transition_weight, context_emission_weight):


    emissionProbs = {}
# with open('../model/emissionProbs.txt', 'r') as infile:
    for line in in_emission_probs.readlines():
        result = globalexpr.match(line)
        tag = result.group(1)
        id = result.group(2)
        prob = float(result.group(3))
        if tag not in emissionProbs:
            emissionProbs[tag] = {}
        emissionProbs[tag][id] = prob


    transitionProbs = {}
# with open('../model/transitionProbs.txt', 'r') as infile:
    for line in in_transition_probs.readlines():
        result = globalexpr.match(line)
        taga = result.group(1)
        tagb = result.group(2)
        prob = float(result.group(3))
        transitionProbs[(taga, tagb)] = prob
    # outfile.write(f"{taga} {tagb}: {transitionProbs[(taga,tagb)]}\n")

    contextTransitionProbs = {}
# with open('../model/contextTransitionProbs.txt', 'r') as infile:
    for line in in_context_transition_probs.readlines():
        result = contextexpr.match(line)
        context = result.group(1)
        taga = result.group(2)
        tagb = result.group(3)
        prob = float(result.group(4))
        if context not in contextTransitionProbs:
            contextTransitionProbs[context] = {}
        contextTransitionProbs[context][(taga, tagb)] = prob
        # outfile.write(f"{context} {taga} {tagb}: {contextTransitionProbs[context][(taga,tagb)]}\n")

    contextemissionProbs = {}
# with open('../model/contextEmissionProbs.txt', 'r') as infile:
    for line in in_context_emission_probs.readlines():
        result = contextexpr.match(line)
        context = result.group(1)
        tag = result.group(2)
        id = result.group(3)
        prob = float(result.group(4))
        if context not in contextemissionProbs:
            contextemissionProbs[context] = {}
        if tag not in contextemissionProbs[context]:
            contextemissionProbs[context][tag] = {}
        contextemissionProbs[context][tag][id] = prob
        # outfile.write(f"{context} {tag} {id}: {contextemissionProbs[context][tag][id]}\n")


    # Viterbi
    def runViterbi(identifier, context):
        identifier = identifier.copy()
        identifier.insert(0,'BOI')
        identifier.append('EOI')

        ## Fill in some value for every state. But sometimes the emission probability will be 0 (e.g., 
        ## the probability that you see "the" when the tag is NN). Account for this possibility *and* for the 
        ## possibility of an OOV, by assigning defaultprob if the emission probability is 0.
        
        defaultprob =  0.00000000001

        ## Create the trellis and backpointer itself as a 2D np array
        stateprobs = np.zeros(shape=(len(identifier), len(common.tags)))
        backpointers = np.zeros(shape=(len(identifier), len(common.tags)))

        ## Initialize state 0, which must be the beginning of sentence (BOS) tag with probability 1
        stateprobs[0,common.tags.index("SOI")] = 1.0

        ## Fill in the rest of the trellis, starting with state 1
        ## Populate both state probabilities and backpointers

        for i in range(1, len(identifier)):
            #for each pos
            for (posIdx, pos) in enumerate(common.tags):
                #calc argmax
                args = [ () for i in range(len(common.tags))]
                #for each prev pos
                for (prevIdx, prevPos) in enumerate(common.tags):
                    #get the prev final pßrob
                    prevProb = stateprobs[i-1, prevIdx]
                    #get the transition prob
                    contexttprob = contextTransitionProbs[context][prevPos, pos] if (prevPos, pos) in contextTransitionProbs[context] else defaultprob
                    globaltprob = transitionProbs[prevPos, pos] if (prevPos, pos) in transitionProbs else defaultprob
                    tprob = ((contexttprob * context_transition_weight) + (globaltprob * global_transition_weight)) / 2
                    #calculate the prev prob * transition and save it with the pos
                    args[prevIdx] = (prevProb * tprob, prevPos)
                    
                #get the max pos and value from prev
                argmax = max(args, key= lambda x: x[0])
                #get emission probablity for the context
                contextWord = identifier[i] #if insent[i] in contextemissionProbs[context][pos] else 'UNK'
                if pos in contextemissionProbs[context] and contextWord in contextemissionProbs[context][pos] and contextemissionProbs[context][pos][contextWord] != 0:
                    contextEmitProb = contextemissionProbs[context][pos][contextWord]
                else:
                    contextEmitProb = common.defaultProb
                #get emission probablility globally
                globalWord = identifier[i] #if insent[i] in contextemissionProbs[context][pos] else 'UNK'
                if pos in emissionProbs and globalWord in emissionProbs[pos] and emissionProbs[pos][globalWord] != 0:
                    globalEmitProb = emissionProbs[pos][globalWord]
                else:
                    globalEmitProb = common.defaultProb
                #calculate the final emit prob as the average of the context prob and the global prob
                emitProb = ((contextEmitProb * context_emission_weight) + (globalEmitProb * global_emission_weight))/2
                #calculate the final probablility
                finalprob = argmax[0] * emitProb
                #record results
                backpointers[i, posIdx] = common.tags.index(argmax[1])
                stateprobs[i,posIdx] = finalprob



        ## Store the string of POS tags in a variable called posseq
        posseq = "EOI"

        ## After populating the full trellis and backpointer trellis, print out the proposed POS tag sequence by traversing
        ## the backpointers from the backpointer in the last column that corresponds to "EOS" (end of sentence tag).

        ## Getting started
        #start at the EOI tag
        maxprob = stateprobs[len(identifier)-1,common.tags.index("EOI")]
        maxprobid = backpointers[len(identifier)-1,common.tags.index("EOI")]

        for i in range(len(identifier)-2, 0, -1):
            # add the maxprob pos to posseq
            posseq = common.tags[maxprobid.astype(int)]+ " " + posseq
            # calculate the next maxprobid
            maxprobid = backpointers[i, maxprobid.astype(int)]
        posseq = "BOI " + posseq
        
        return posseq.split()[1:-1]

    return runViterbi