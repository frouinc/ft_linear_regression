#!/usr/bin/env python3

import sys
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import argparse
import progressbar

def verb(str):
	if (verbose):
		print(str)

def calculateStart():
	a = (dataset[1][dataSize - 1] - dataset[1][0]) / (dataset[0][dataSize - 1] - dataset[0][0])
	b = dataset[1][0] - a * dataset[0][0]
	t[0] = b
	t[1] = a

def calculate():
	tmp0 = t[0] - a * (1. / dataSize) * sum0()
	tmp1 = t[1] - a * (1. / dataSize) * sum1()
	t[0] = tmp0
	t[1] = tmp1
	if (math.isnan(t[0]) or math.isnan(t[1]) or math.isinf(t[0]) or math.isinf(t[1])):
		return True
	return False

parser = argparse.ArgumentParser(description='Calculate the values for the linear regression of a 2D data set.')
# File
parser.add_argument('filename',
	help = 'File containing the dataset to learn from.')
# Iterations
parser.add_argument('-i', '--iterations',
	metavar = 'N',
	type = int,
	default = 1000000,
	help = 'How many iterations will the algorithm go through.')
# Step
parser.add_argument('-s', '--step',
	metavar = 'N',
	type = float,
	default = 0.01,
	help = 'Size of the step to take at each iteration.')
# Verbose
parser.add_argument('-v', '--verbose',
	action = 'store_true',
	help = 'Display some verbose')
# Graph
parser.add_argument('-g', '--graph',
	action = 'store_true',
	help = 'Display a graph with the data at the end')

args = parser.parse_args()

t = [0., 0.]
a = args.step
iterations = args.iterations
verbose = args.verbose
graph = args.graph
filename = args.filename

verb('Step has been set to : ' + str(a))
verb('Iterations has been set to : ' + str(iterations))
verb('Filename has been set to : ' + filename)

# Get dataset
dataset = [[], []]
try:
	with open(filename) as csvDataFile:
		verb('Opening and reading ' + filename + '.')
		csvReader = csv.reader(csvDataFile)
		next(csvReader, None)
		for row in csvReader:
			dataset[0].append(float(row[0]))
			dataset[1].append(float(row[1]))
		verb('Done reading file')
except IOError:
	print('Could not open given file, please verify that you have read access on it.')
	sys.exit()
dataSize = len(dataset[0])
verb('Dataset has ' + str(dataSize) + ' values')
calculateStart()

# Calculate plotting range
plotRange = [[min(dataset[0]), max(dataset[0])], [min(dataset[1]), max(dataset[1])]]
xRange = abs(plotRange[0][0] - plotRange[0][1])
yRange = abs(plotRange[1][0] - plotRange[1][1])
plotRange[0][0] -= (xRange / 10)
plotRange[0][1] += (xRange / 10)
plotRange[1][0] -= (yRange / 10)
plotRange[1][1] += (yRange / 10)

def h(x):
	return (t[0] + t[1] * x)

def sum0():
	result = 0.0
	for i in range(0, dataSize):
		result += (h(dataset[0][i]) - dataset[1][i])
	return result

def sum1():
	result = 0.0
	for i in range(0, dataSize):
		result += ((h(dataset[0][i]) - dataset[1][i]) * dataset[0][i])
	return result


verb('Starting to calculate the linear regression with ' + str(iterations) + ' iterations.')
if verbose:
	with progressbar.ProgressBar(max_value=iterations) as bar:
		i = 0
		while i < iterations:
			if (calculate()):
				i = -1
				a *= 0.9
				calculateStart()
				verb('\nStep was too big, retrying with step = ' + str(a))
			else:
				bar.update(i)
			i += 1
else:
	i = 0
	while i < iterations:
		if (calculate()):
			i = -1
			a *= 0.9
			calculateStart()
		i += 1
verb('Done calculating the linear regression.')

verb('Results of the linear regression:')
verb('t0 = ' + str(t[0]))
verb('t1 = ' + str(t[1]))

if graph:
	verb('Plotting dataset.')
	plt.plot(dataset[0], dataset[1], 'ro')
	plt.axis([plotRange[0][0], plotRange[0][1], plotRange[1][0], plotRange[1][1]])
	rr = np.arange(plotRange[0][0] - 10, plotRange[0][1] + 10, 1)
	plt.plot(rr, h(rr).astype(np.float))
	plt.show()

verb('Writing the data inside .theta file')
saveFile = open('.theta', 'w')
saveFile.write(str(t[0]) + ',' + str(t[1]))
saveFile.close()