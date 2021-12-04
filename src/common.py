defaultProb = 0.000001
unkThreshold = 1

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
