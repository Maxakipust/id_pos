from flask import Flask
import evaluate_pos
import clean_ids

app = Flask(__name__)

@app.route("/<type>/<name>/<context>")
def tag_id(type, name, context):
    nameArr = clean_ids.camel_case_split(name).split(" ")
    posStr = evaluate_pos.runViterbi(nameArr, context).split()
    posStr = posStr[1:-1]
    result = ",".join(list(map(lambda a: f"{a[0]}|{a[1]}", zip(nameArr, posStr))))
    print(name, result)
    return result