import csv
lines = []
with open('data/unlabeled_ids.txt', 'a') as outfile:
    with open('data/orig_training_data.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            line = row['IDENTIFIER'].lower()
            if line not in lines:
                outfile.write(line + "\n")
                lines.append(line)
            