from __future__ import print_function
import sys

def read_predictions(filename):
    """ Read predictions into dictionary"""
    d = {}
    with open(filename) as f:
        for line in f:
           (key, val) = line.split()
           d[key] = val
    return d


# Read in arguments
ground_truth_file = sys.argv[1]
results_file = sys.argv[2]

print( "Results file " + results_file )
print( "Groundtruth file " + ground_truth_file )

results_map = read_predictions(results_file)
ground_truth_map = read_predictions(ground_truth_file)

# Calculate accuracy and print incorrect predictions
correct = 0
for ID in ground_truth_map:
    label = ground_truth_map[ID]
    if ID not in results_map:
        print( "Missing predictions for " + ID)
    elif results_map[ID] == label:
        correct = correct + 1
    else:
        print( "Incorrect " + ID )

# Print summary
print( str(correct) + " out of " + str(len(ground_truth_map)) + " were correct!")
print( "accuracy " + str(float(correct)/len(ground_truth_map)))