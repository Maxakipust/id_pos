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


def print_confusion(confusion):
    print(pd.DataFrame(confusion, 
        index=list(map((lambda x: 'true:'+x), common.used_tags)), #['true:yes', 'true:no'], 
        columns=list(map((lambda x: 'pred:'+x), common.used_tags))#['pred:yes', 'pred:no']
    ))

noun_tags = ["N", "NM", "NPL"]
verb_tags = ["V", "VM"]

wnl = WordNetLemmatizer()
def isplural(word):
    lemma = wnl.lemmatize(word, 'n')
    return True if word is not lemma else False

p = inflect.engine()
def new_tagger(id, context, evaluate):
    # print("running new tagger")
    actual_result = evaluate(id, context)
    # print("actual result:")
    # print(actual_result)
    for (index, pos) in enumerate(actual_result):
        next_pos = actual_result[index + 1] if index < len(actual_result) - 1 else ""
        # if pos in noun_tags:
        #     if next_pos in noun_tags:
        #         actual_result[index] = "NM"
        #     elif pos == "NM":
        #         if not isplural(id[index]):
        #             actual_result[index] = "N"
        #         else:
        #             actual_result[index] = "NPL"
        if pos in verb_tags:
            if next_pos in verb_tags:
                actual_result[index] = "VM"
            else:
                actual_result[index] = "V"

    # print("final result")
    # print(actual_result)
    return actual_result
    

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
    new_tag_fn = lambda id, context: new_tagger(id, context, globaltagFn)
    # print(new_tag_fn(['validate', 'instance', 'methods'], "FUNCTION"))
    (confusion, report) = test_model.test_model(new_tag_fn, orig_test_data)
    print_confusion(confusion)
    print(report)
    

def full_run():
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


def find_optimal_weights():
    results = {}
    orig_training_data = open("data/orig_training_data.csv", "r")
    orig_test_data = open("data/orig_unseen_testing_data.csv", "r")
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

    for global_weight_emission in range(0,100,10):
        for global_weight_transition in range(0,100,10):
            context_weight_emission = 100 - global_weight_emission
            context_weight_transition = 100 - global_weight_transition
            tag_fn = evaluate_pos.load_probs(aug_hmm_global_emission_probs, aug_hmm_global_transition_probs, aug_hmm_context_emission_probs, aug_hmm_context_transition_probs,
            global_weight_transition/100.0, global_weight_emission/100.0, context_weight_transition/100.0, context_weight_emission/100.0)
            new_tag_fn = lambda id, context: new_tagger(id, context, tag_fn)
            (confusion, report) = test_model.test_model(new_tag_fn, orig_test_data)
            results[(global_weight_emission, global_weight_transition)] = report["accuracy"]
    print("global_emission_weight, global_transition_weight, context_emission_weight, context_transition_weight, accuracy")
    for (global_weight_emission, global_weight_transition) in results:
        print(f"{global_weight_emission}, {global_weight_transition}, {100 - global_weight_emission}, {100 - global_weight_transition}, {results[(global_weight_emission, global_weight_transition)]}")

def run_baseline(tag):
    baseline_fn = lambda identifier, c: [tag for w in identifier]
    orig_test_data = open("data/orig_unseen_testing_data.csv", "r")
    (confusion, report) = test_model.test_model(baseline_fn, orig_test_data)
    print_confusion(confusion)
    print(report)

# run_baseline("NM")
run_new_tagger()
# find_optimal_weights()
# full_run()