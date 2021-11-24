
def camel_case_split(str):
    words = [[str[0]]]
  
    for c in str[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append(list(c))
        elif words[-1][-1].isnumeric() != c.isnumeric():
            words.append(list(c))
        else:
            words[-1].append(c)
  
    return ' '.join([' '.join(''.join(word).lower().split('_')) for word in words])


ids = []
with open('data/raw_ids.txt', 'r') as infile:
    with open('data/unlabeled_ids.txt', 'a+') as outfile:
        for inline in infile:
            if inline not in ids:
                ids.append(inline)
                final = camel_case_split(inline)
                if len(final) > 3:
                    outfile.write(final)
