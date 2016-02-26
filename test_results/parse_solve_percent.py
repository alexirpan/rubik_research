import glob

# This is horribly horribly hardcoded all over the place, but this should
# only ever be run once so whatever
files = glob.glob('models_*')

scores = dict()
for name in files:
    with open(name) as f:
        contents = f.read()
        contents = contents.split('Solved ')[1]
        contents = contents.split('%')[0]
        scores[name] = float(contents)

print scores

import csv
with open('scores.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['name', 'percent'])
    items = sorted(scores.items())
    for row in items:
        writer.writerow(row)

