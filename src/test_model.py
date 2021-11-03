import evaluate_pos
import csv
import common
import requests
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



#run tests
with open('data/orig_unseen_testing_data.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    prevId = ""
    nltkMap = {}
    for row in reader:
        if row['IDENTIFIER'] != prevId:
            prevId = row['IDENTIFIER']
            idArr = row['IDENTIFIER'].lower().split()
            idArr = list(map(common.cleanUpWord, idArr))
            print(idArr)
            context = common.contexts[int(row['CONTEXT']) - 1]
            calcPOS = evaluate_pos.runViterbi(idArr, context).split()[1:-1]
            ensemblePOS = runEnsemble("int", idArr.join(), context)

            actualPOS = row['GRAMMAR_PATTERN'].split()
            print("actual", actualPOS)

            print("calc", calcPOS)
            for (index, actual) in enumerate(actualPOS):
                calc = calcPOS[index]
                if calc == actual:
                    success += 1
                else:
                    fail += 1
                    print("fail", actual, calc)
            print()
                    

print("success",success)
print("fail", fail)
print("acc",success/(success+fail))

# print("nltk_success",nltk_success)
# print("nltk_fail", nltk_fail)
# print("nltk_acc",nltk_success/(nltk_success+nltk_fail))