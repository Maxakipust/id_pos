from flask import Flask
import evaluate_pos
import clean_ids

app = Flask(__name__)
#to start webserver cd src, run export FLASK_APP=hello, flask run
#small webserver to expose the tagger
base_hmm_global_emission_probs = open("../model/baseHMM/global_emission_probs.txt", "r+")
base_hmm_global_transition_probs = open("../model/baseHMM/global_transition_probs.txt", "r+")
base_hmm_context_emission_probs = open("../model/baseHMM/context_emission_probs.txt", "r+")
base_hmm_context_transition_probs = open("../model/baseHMM/context_transition_probs.txt", "r+")
tag_fn = evaluate_pos.load_probs(base_hmm_global_emission_probs, base_hmm_global_transition_probs, base_hmm_context_emission_probs, base_hmm_context_transition_probs,
0.5,0.5,0.5,0.5)


@app.route("/<type>/<name>/<context>")
def tag_id(type, name, context):
    name = name.replace("()", "")
    nameArr = clean_ids.camel_case_split(name).split(" ")
    posStr = tag_fn(nameArr, context)
    result = ",".join(list(map(lambda a: f"{a[0]}|{a[1]}", zip(nameArr, posStr))))
    print(name, result)
    return result