from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from numpy.core.numeric import full
import generate_probs
import test_model
import evaluate_pos
import common
import pandas as pd
import augment_labeled_data
import inflect
import word2vec_clustering 
import graph_clustering
import sys


#prints a confusion matrix with labels
def print_confusion(confusion):
    print(pd.DataFrame(confusion, 
        index=list(map((lambda x: 'true:'+x), common.used_tags)), #['true:yes', 'true:no'], 
        columns=list(map((lambda x: 'pred:'+x), common.used_tags)) #['pred:yes', 'pred:no']
    ))

noun_tags = ["N", "NM", "NPL"]
verb_tags = ["V", "VM"]

#determine if a word is plural. It doesn't always work, but its ok
wnl = WordNetLemmatizer()
def isplural(word):
    lemma = wnl.lemmatize(word, 'n')
    return True if word is not lemma else False

# perform post processing on an existing tagger
# finds sequences of noun like tags, makes all of them NM except for the head noun
p = inflect.engine()
def post_process(id, context, evaluate):
    actual_result = evaluate(id, context)
    for (index, pos) in enumerate(actual_result):
        next_pos = actual_result[index + 1] if index < len(actual_result) - 1 else ""
        if pos in noun_tags:
            if next_pos in noun_tags:
                actual_result[index] = "NM"
            elif pos == "NM":
                if not isplural(id[index]):
                    actual_result[index] = "N"
                else:
                    actual_result[index] = "NPL"
        if pos in verb_tags:
            if next_pos in verb_tags:
                actual_result[index] = "VM"
            else:
                actual_result[index] = "V"

    # print("final result")
    # print(actual_result)
    return actual_result
    
# run the postprocessing on the base HMM
def run_new_tagger():
    print("running base HMM with only global probs")
    orig_training_data = open("data/orig_training_data.csv", "r")
    orig_test_data = open("data/orig_unseen_testing_data.csv", "r")
    base_hmm_global_emission_probs = open("model/baseHMM/global_emission_probs.txt", "w+")
    base_hmm_global_transition_probs = open("model/baseHMM/global_transition_probs.txt", "w+")
    base_hmm_context_emission_probs = open("model/baseHMM/context_emission_probs.txt", "w+")
    base_hmm_context_transition_probs = open("model/baseHMM/context_transition_probs.txt", "w+")
    generate_probs.generate_probabilities(orig_training_data, base_hmm_global_emission_probs, base_hmm_global_transition_probs, base_hmm_context_emission_probs, base_hmm_context_transition_probs)
    
    print("calculated probabilites")
    globaltagFn = evaluate_pos.load_probs(base_hmm_global_emission_probs, base_hmm_global_transition_probs, base_hmm_context_emission_probs, base_hmm_context_transition_probs,
    0.5, 0.5, 0.5, 0.5)
    new_tag_fn = lambda id, context: post_process(id, context, globaltagFn)
    # print(new_tag_fn(['validate', 'instance', 'methods'], "FUNCTION"))
    (confusion, report) = test_model.test_model(new_tag_fn, orig_test_data)
    print_confusion(confusion)
    print(report)
    
#runs a bunch of different configs and prints out the results 
def full_run():
    long = True if len(sys.argv) > 1 and sys.argv[1] == "long" else False
    print("running base HMM with only global probs")
    orig_training_data = open("data/orig_training_data.csv", "r")
    orig_test_data = open("data/orig_unseen_testing_data.csv", "r")
    base_hmm_global_emission_probs = open("model/baseHMM/global_emission_probs.txt", "w+")
    base_hmm_global_transition_probs = open("model/baseHMM/global_transition_probs.txt", "w+")
    base_hmm_context_emission_probs = open("model/baseHMM/context_emission_probs.txt", "w+")
    base_hmm_context_transition_probs = open("model/baseHMM/context_transition_probs.txt", "w+")
    generate_probs.generate_probabilities(orig_training_data, base_hmm_global_emission_probs, base_hmm_global_transition_probs, base_hmm_context_emission_probs, base_hmm_context_transition_probs)
    
    print("calculated probabilites")
    globaltagFn = evaluate_pos.load_probs(base_hmm_global_emission_probs, base_hmm_global_transition_probs, base_hmm_context_emission_probs, base_hmm_context_transition_probs,
    1.0, 1.0, 0.0, 0.0)
    (confusion, report) = test_model.test_model(globaltagFn, orig_test_data)
    print_confusion(confusion)
    print(report)

    print("running base HMM with only context probs")
    contexttagFn = evaluate_pos.load_probs(base_hmm_global_emission_probs, base_hmm_global_transition_probs, base_hmm_context_emission_probs, base_hmm_context_transition_probs,
    0.0, 0.0, 1.0, 1.0)
    (confusion, report) = test_model.test_model(contexttagFn, orig_test_data)
    print_confusion(confusion)
    print(report)

    print("running base HMM with context and global probs")
    bothtagFn = evaluate_pos.load_probs(base_hmm_global_emission_probs, base_hmm_global_transition_probs, base_hmm_context_emission_probs, base_hmm_context_transition_probs,
    0.5, 0.5, 0.5, 0.5)
    (confusion, report) = test_model.test_model(bothtagFn, orig_test_data)
    print_confusion(confusion)
    print(report)
    print()

    print("augmenting training data")
    
    augmented_syn_training_data = open('model/augmented/aug_syn_training_data.csv', 'w+')
    augment_labeled_data.syn_labeled_data(orig_training_data, augmented_syn_training_data)
    augmented_pl_training_data = open('model/augmented/aug_pl_training_data.csv', 'w+')
    print()
    augment_labeled_data.plural_labeled_data(augmented_syn_training_data, augmented_pl_training_data)
    aug_hmm_global_emission_probs = open("model/augmented/global_emission_probs.txt", "w+")
    aug_hmm_global_transition_probs = open("model/augmented/global_transition_probs.txt", "w+")
    aug_hmm_context_emission_probs = open("model/augmented/context_emission_probs.txt", "w+")
    aug_hmm_context_transition_probs = open("model/augmented/context_transition_probs.txt", "w+")
    generate_probs.generate_probabilities(augmented_pl_training_data, aug_hmm_global_emission_probs, aug_hmm_global_transition_probs, aug_hmm_context_emission_probs, aug_hmm_context_transition_probs)

    print("running augmented HMM with context and global probs")
    bothtagFn = evaluate_pos.load_probs(aug_hmm_global_emission_probs, aug_hmm_global_transition_probs, aug_hmm_context_emission_probs, aug_hmm_context_transition_probs,
    0.5, 0.5, 0.5, 0.5)
    (confusion, report) = test_model.test_model(bothtagFn, orig_test_data)
    print_confusion(confusion)
    print(report)


    untagged_ids = open("data/unlabeled_ids.txt", "r+")
    print("running augmented emission data with word2vec")
    # word2vec_model = open("model/word2vec/word2vec.model", "w+")
    word2vec_probs = None
    if long:
        print("This will take a long time. to disable run 'main.py'")
        word2vec_probs = open("model/word2vec/global_emission_probs.txt", "w+")
        word2vec_clustering.word2vec_clustering(untagged_ids, base_hmm_global_emission_probs, "model/word2vec/word2vec.model", word2vec_probs)
    else:
        word2vec_probs = open("model/word2vec/global_emission_probs.txt", "r+")
        print("not building new word2vec model, to enable run 'main.py long'")
    word2vec_tag_fn = evaluate_pos.load_probs(word2vec_probs, base_hmm_global_transition_probs, base_hmm_context_emission_probs, base_hmm_context_transition_probs, 
    0.5, 0.5, 0.5, 0.5)
    (confusion, report) = test_model.test_model(word2vec_tag_fn, orig_test_data)
    print_confusion(confusion)
    print(report)

    print("running augmented emission data with neighbor graph clustering")
    # graph = open("model/graph/graph.gexf", "w+")
    graph_probs = None
    if long:
        print("This will take a long time. To disable run 'main.py'")
        graph_probs = open("model/graph/global_emission_probs.txt", "w+")
        graph_clustering.augment_emission_probs_with_custom_clustering(untagged_ids,base_hmm_global_emission_probs , "model/graph/graph.gexf", graph_probs)
    else:
        graph_probs = open("model/graph/global_emission_probs.txt", "r+")
        print("not building new graph clustering probabilities, to enable run 'main.py long'")
    graph_tag_fn = evaluate_pos.load_probs(graph_probs, base_hmm_global_transition_probs, base_hmm_context_emission_probs, base_hmm_context_transition_probs, 
    0.5, 0.5, 0.5, 0.5)
    (confusion, report) = test_model.test_model(graph_tag_fn, orig_test_data)
    print_confusion(confusion)
    print(report)

#runs the base HMM with a bunch of different weights and prints the result.
#note in order to use this, you should make sure to add output_dict=True to the classification report in test_model
def find_optimal_weights():
    results = {}
    orig_training_data = open("data/orig_training_data.csv", "r")
    orig_test_data = open("data/orig_unseen_testing_data.csv", "r")
    
    base_hmm_global_emission_probs = open("model/baseHMM/global_emission_probs.txt", "w+")
    base_hmm_global_transition_probs = open("model/baseHMM/global_transition_probs.txt", "w+")
    base_hmm_context_emission_probs = open("model/baseHMM/context_emission_probs.txt", "w+")
    base_hmm_context_transition_probs = open("model/baseHMM/context_transition_probs.txt", "w+")
    generate_probs.generate_probabilities(orig_training_data, base_hmm_global_emission_probs, base_hmm_global_transition_probs, base_hmm_context_emission_probs, base_hmm_context_transition_probs)

    for global_weight_emission in range(-100,100,10):
        for global_weight_transition in range(-100,100,10):
            context_weight_emission = 100 - global_weight_emission
            context_weight_transition = 100 - global_weight_transition
            tag_fn = evaluate_pos.load_probs(base_hmm_global_emission_probs, base_hmm_global_transition_probs, base_hmm_context_emission_probs, base_hmm_context_transition_probs,
            global_weight_transition/100.0, global_weight_emission/100.0, context_weight_transition/100.0, context_weight_emission/100.0)
            new_tag_fn = lambda id, context: post_process(id, context, tag_fn)
            (confusion, report) = test_model.test_model(new_tag_fn, orig_test_data)
            results[(global_weight_emission, global_weight_transition)] = report["accuracy"]
    print("global_emission_weight, global_transition_weight, context_emission_weight, context_transition_weight, accuracy")
    for (global_weight_emission, global_weight_transition) in results:
        print(f"{global_weight_emission}, {global_weight_transition}, {100 - global_weight_emission}, {100 - global_weight_transition}, {results[(global_weight_emission, global_weight_transition)]}")

#run a baseline against the test set
def run_baseline(tag):
    baseline_fn = lambda identifier, c: [tag for w in identifier]
    orig_test_data = open("data/orig_unseen_testing_data.csv", "r")
    (confusion, report) = test_model.test_model(baseline_fn, orig_test_data)
    print_confusion(confusion)
    print(report)

# run_baseline("NM")
# run_new_tagger()
# find_optimal_weights()
full_run()