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

# test a model against the test data
def test_model(tag_id_fn, test_file):
    test_file.seek(0)
    reader = csv.DictReader(test_file)
    prevId = ""
    calculated_total = []
    actual_total = []
    with_unk = False
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
                if calc == "UNK" or actual == "UNK":
                    with_unk = True
                # if not calc == actual:
                    # print(idArr, context)
                    # print(calcPOS)
                    # print(actualPOS)
                    # print()
    if with_unk:
        confusion = confusion_matrix(actual_total, calculated_total, labels=common.with_unk)
        report = classification_report(actual_total, calculated_total, labels=common.with_unk) #, output_dict=True)
    else:
        confusion = confusion_matrix(actual_total, calculated_total, labels=common.used_tags)
        report = classification_report(actual_total, calculated_total, labels=common.used_tags) #, output_dict=True)
    return (confusion, report)