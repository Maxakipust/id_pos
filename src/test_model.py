import evaluate_pos
import csv
import common
import requests
from sklearn.metrics import confusion_matrix, classification_report
# import nltk

success = 0
fail = 0

def runEnsemble(type, name, context):
    try:
        name = "_".join(name)
        if context == "FUNCTION":
            name += "()"
        response = requests.get(f"http://localhost:5000/{type}/{name}/{context}")
        print("response", response.text)
        response = response.text.split(',')
        result = []
        for section in response:
            result.append(section.split('|')[1])
        return " ".join(result)
    except:
        return ""

# def runNLTK(name):
#     pos = nltk.pos_tag(name)
#     return list(map(lambda arg: arg[1], pos))

def test_model(tag_id_fn, test_file):
    reader = csv.DictReader(test_file)
    prevId = ""
    calculated_total = []
    actual_total = []
    for row in reader:
        if row['IDENTIFIER'] != prevId:
            prevId = row['IDENTIFIER']
            idArr = row['IDENTIFIER'].lower().split()
            idArr = list(map(common.cleanUpWord, idArr))
            context = common.contexts[int(row['CONTEXT']) - 1]
            calcPOS = tag_id_fn(idArr, context)
            actualPOS = row['GRAMMAR_PATTERN'].split()
            for (calc, actual) in list(zip(calcPOS, actualPOS)):
                calculated_total.append(calc)
                actual_total.append(actual)
    confusion = confusion_matrix(actual_total, calculated_total, labels=common.used_tags)
    report = classification_report(actual_total, calculated_total, labels=common.used_tags)
    return (confusion, report)