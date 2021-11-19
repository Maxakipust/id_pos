import csv
with open('data/unlabeled_ids.txt', 'a') as outfile:
    with open('data/orig_training_data.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            outfile.write(row['IDENTIFIER'].lower() + "\n")
            