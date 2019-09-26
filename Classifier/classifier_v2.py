import numpy as np 
import functools
import csv
import collections
import re
import os
import glob
import spacy 
import operator

nlp = spacy.load("en_core_web_sm")

vocabulary = {}
vocabulary_25plus = {}
tot_number_pos=0
tot_number_neg=0

class VocItem:
	def __init__(self,word):
		self.word = word
		self.p_count = 0
		self.n_count = 0
		self.all_count = 0

print("\n ------- \n")

# Tokenising all text files in train set to build vocabulary
def get_tokens(files):
    doc_tokens = []
    for file in files:
        doc = open(file, "r")
        if doc.mode == "r":
            tokens = [token.text.lower() for token in nlp(doc.read())]
            for token in tokens:
            	if re.search("[a-zA-Z0-9]", token):
            		doc_tokens.append(token)

    return doc_tokens

#Creating vocabulary from the train set
def CreateVocabulary():
	files_pos = glob.glob('./movies/train/P*.txt')
	files_neg = glob.glob('./movies/train/N*.txt')
	tokens_pos = get_tokens(files_pos)
	tokens_neg = get_tokens(files_neg)

	for token in tokens_pos:
		if token not in vocabulary:
			vocabulary[token] = VocItem(token)		

		vocabulary[token].p_count += 1
		vocabulary[token].all_count += 1
		global tot_number_pos
		tot_number_pos += 1

	for token in tokens_neg:
		if token not in vocabulary:
			vocabulary[token] = VocItem(token)		

		vocabulary[token].n_count += 1
		vocabulary[token].all_count += 1
		global tot_number_neg
		tot_number_neg += 1

	for key in vocabulary:
		if vocabulary[key].all_count > 24:
			vocabulary_25plus[key] = vocabulary[key]

	#print vocabulary
	#for key in vocabulary_25plus:
		#print(vocabulary_25plus[key].word +" - "+str(vocabulary_25plus[key].p_count)+" - "+str(vocabulary_25plus[key].n_count)+" - "+str(vocabulary_25plus[key].all_count)+"\n")

#get tokens of one file (used for test set)
def tokenize(file):
	doc_tokens = []
	#print(file[0])
	doc = open(file[0], "r")
	tokens = [token.text.lower() for token in nlp(doc.read())]

	for token in tokens:
		if re.search("[a-zA-Z0-9]", token):
			doc_tokens.append(token)
	return doc_tokens

pos = glob.glob('./movies/test/P-test49.txt')


# --------- SMOOTHED PROBABILITY FOR ONE WORD ----------------#
# P(wi|Pos) ≈ C(wi,Pos) + k / NPos +kV
# C(wi,Pos) is the frequency of word wi in the positive reviews
# NPos = total number of words in the positive reviews
# V = size of vocabulary
# k = user defined smoothing parameter
# w = word you search for 
# rating = positive or negative review (p or n)
def smoothProb(w , rating , k=1):
	
	#if w in vocabulary_25plus:
	if rating == "p":
		smoothProb = (vocabulary_25plus[w].p_count + k) / (tot_number_pos + k* len(vocabulary_25plus))

	if rating == "n":
		smoothProb = (vocabulary_25plus[w].n_count + k) / (tot_number_neg + k* len(vocabulary_25plus))

	return smoothProb

# --------- PROBABILITY OF LIST OF WORDS IN LOG SPACE----------------#
# logP(w1, w2, w3 ... wn) = sum(log(P(wi)))

def wListProbLog(wList , rating , k):
	prop = 0
	for w in wList:
		if w in vocabulary_25plus:
			prop = prop + np.log(smoothProb(w,rating,k))

	return prop

# --------- PREDICT CLASS FOR EACH REVIEW IN TEST SET ----------------#
def classifyReviews(k):
	test_files = glob.glob('./movies/test/*.txt')
	review_list = []
	rating=""

	result =open("result.txt","w+")
	for test_file in test_files:
		positive = 0
		negative = 0
		filename = os.path.splitext(test_file)[0].split("/")[3]
		review_list.append(filename)
		tokens = tokenize(glob.glob(test_file))
		positive = wListProbLog(tokens,"p",k)
		negative = wListProbLog(tokens,"n",k)


		if positive > negative:
			rating =  "P"
		if positive < negative:
			rating =  "N"

		result.write(filename+"\t"+rating+"\n")


	result.close()

# --------- TEST THE FUCNTIONS ----------------#

CreateVocabulary()

classifyReviews(1)









