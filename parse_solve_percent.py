"""Collects test results into one CSV file.

This assumes many, many things about the structure of the
file outputs. It's bad style but it works.
"""
import argparse
parser = argparse.ArgumentParser(
    description='Collects output into CSV. Given a directory, '
    'it finds the solve percentage in test output files and collates'
    'them together into a CSv dropped into the same directory'
)
parser.add_argument('results_dir', help='Results directoy')
args = parser.parse_args()

names = [args.results_dir + '/' + str(i) for i in range(1, 26 + 1)]

scores = {}
for i, name in enumerate(names):
    with open(name) as f:
        contents = f.read()
        if 'no cubes' in contents:
            scores[i+1] = 0
            continue
        contents = contents.split('Solved ')[1]
        contents = contents.split('%')[0]
        scores[i+1] = float(contents)

print scores

import csv
with open(args.results_dir + '/scores.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['name', 'percent'])
    items = sorted(scores.items())
    for row in items:
        writer.writerow(row)

