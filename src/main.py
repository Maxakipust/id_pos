from numpy.core.numeric import full
import generate_probs
import test_model
import evaluate_pos
import common
import pandas as pd

all_files = []

def print_confusion(confusion):
    print(pd.DataFrame(confusion, 
        index=list(map((lambda x: 'true:'+x), common.used_tags)), #['true:yes', 'true:no'], 
        columns=list(map((lambda x: 'pred:'+x), common.used_tags))#['pred:yes', 'pred:no']
    ))


def reset_all_files(files):
    for f in files:
        f.seek(0)


def full_run():
    print("running base HMM with only global probs")
    orig_training_data = open("data/orig_training_data.csv", "r")
    orig_test_data = open("data/orig_unseen_testing_data.csv", "r")
    base_hmm_global_emission_probs = open("model/baseHMM/global_emission_probs.txt", "a+")
    base_hmm_global_transition_probs = open("model/baseHMM/global_transition_probs.txt", "a+")
    base_hmm_context_emission_probs = open("model/baseHMM/context_emission_probs.txt", "a+")
    base_hmm_context_transition_probs = open("model/baseHMM/context_transition_probs.txt", "a+")
    all_files = [orig_training_data, orig_test_data, base_hmm_global_emission_probs, base_hmm_global_transition_probs, base_hmm_context_emission_probs, base_hmm_context_transition_probs]
    reset_all_files(all_files)
    generate_probs.generate_probabilities(orig_training_data, base_hmm_global_emission_probs, base_hmm_global_transition_probs, base_hmm_context_emission_probs, base_hmm_context_transition_probs)
    
    print("calculated probabilites")
    reset_all_files(all_files)
    globaltagFn = evaluate_pos.load_probs(base_hmm_global_emission_probs, base_hmm_global_transition_probs, base_hmm_context_emission_probs, base_hmm_context_transition_probs,
    1.0, 1.0, 0.0, 0.0)
    reset_all_files(all_files)
    (confusion, report) = test_model.test_model(globaltagFn, orig_test_data)
    print_confusion(confusion)
    print(report)

    print("running base HMM with only context probs")
    reset_all_files(all_files)
    contexttagFn = evaluate_pos.load_probs(base_hmm_global_emission_probs, base_hmm_global_transition_probs, base_hmm_context_emission_probs, base_hmm_context_transition_probs,
    0.0, 0.0, 1.0, 1.0)
    reset_all_files(all_files)
    (confusion, report) = test_model.test_model(contexttagFn, orig_test_data)
    print_confusion(confusion)
    print(report)

    print("running base HMM with context and global probs")
    reset_all_files(all_files)
    bothtagFn = evaluate_pos.load_probs(base_hmm_global_emission_probs, base_hmm_global_transition_probs, base_hmm_context_emission_probs, base_hmm_context_transition_probs,
    0.5, 0.5, 0.5, 0.5)
    reset_all_files(all_files)
    (confusion, report) = test_model.test_model(bothtagFn, orig_test_data)
    print_confusion(confusion)
    print(report)

full_run()