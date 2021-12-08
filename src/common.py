
#default probability to use for unknowns or 0s 
defaultProb = 0.000001

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
    'V',
    'VM',
    'PR',
    'D',
    'PRE',
    'EOI'
]
# an array of all the tags we have in the test dataset to quiet warnings when generating reports
used_tags = [
    'N',
    'DT',
    'CJ',
    'P',
    'NPL',
    'NM',
    'V',
    'VM',
    # 'PR',
    'D',
    'PRE'
]

#remove numbers from words
def cleanUpWord(id):
    try:
        idInt = int(id)
        return "NUM"
    except:
        return id
