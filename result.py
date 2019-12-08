#!/usr/bin/env python3

import sys
import csv
import argparse

parser = argparse.ArgumentParser(description='Get the estimated value based on the training from training.py')
parser.add_argument('value',
	type = float,
	help = 'Value to look for.')

args = parser.parse_args()

value = args.value
t = [0., 0.]

try:
	with open('.theta') as datafile:
		csvReader = csv.reader(datafile)
		for row in csvReader:
			t[0] = float(row[0])
			t[1] = float(row[1])
except IOError:
	print('Could not open the .theta file. Please run the training on a dataset first.')
	sys.exit()

result = t[1] * value + t[0]

print(result)