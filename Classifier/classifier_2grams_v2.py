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

bigramVocabulary = {}
tot_number_pos_bigram=0
tot_number_neg_bigram=0



class VocItem:
	def __init__(self,word):
		self.word = word
		self.p_count = 0
		self.n_count = 0
		self.all_count = 0

class BigramItem:
	def __init__(self,bigram):
		self.bigram = bigram
		self.bigram_p_count = 0
		self.bigram_n_count = 0
		self.bigram_all_count = 0
		self.word0 = ""
		self.word1 = ""

print("\n ------- \n")

#get tokens of one file 
def tokenize(file):
	doc_tokens = []
	doc_tokens.append("<s>")
	#print(file[0])
	doc = open(file[0], "r")
	tokens = [token.text.lower() for token in nlp(doc.read())]

	for token in tokens:
		if re.search("[a-zA-Z0-9]", token):
			doc_tokens.append(token)
	return doc_tokens

def get_tokens_bigram(files):
	doc_tokens = []
	for file in files:
		#for every new file add a start of file sign
		doc_tokens.append("<s>")
		doc = open(file, "r")
		if doc.mode == "r":
			tokens = [token.text.lower() for token in nlp(doc.read())]
			for token in tokens:
            	#print(token)
				if re.search("[a-zA-Z0-9]", token):
					doc_tokens.append(token)	

	return doc_tokens


def CreateBigramVocabulary():
	files_pos = glob.glob('./movies/train/P*.txt')
	files_neg = glob.glob('./movies/train/N*.txt')
	bigram_list_pos = []
	bigram_list_neg = []

	# ---- GOING THROUGH THE POSITIVE REVIEWS -----#
	#for file_pos in files_pos:
	tokens_pos = get_tokens_bigram(files_pos)

	#make tokens into pairs
	i = 1
	while i < len(tokens_pos):
		bigram_list_pos.append([tokens_pos[i-1],tokens_pos[i]])
		i += 1


	#load token pairs to a bigram dictionary for easier seach
	i=0
	while i < len(bigram_list_pos):
		bigram = bigram_list_pos[i][0]+"|"+bigram_list_pos[i][1]
		if bigram not in bigramVocabulary:
			bigramVocabulary[bigram] = BigramItem(bigram)
			bigramVocabulary[bigram].word0 = bigram_list_pos[i][0]	
			bigramVocabulary[bigram].word1 = bigram_list_pos[i][1]
		
		bigramVocabulary[bigram].bigram_p_count += 1
		bigramVocabulary[bigram].bigram_all_count += 1	
		global tot_number_pos_bigram
		tot_number_pos_bigram += 1
		i += 1

	# ---- GOING THROUGH THE NEGATIVE REVIEWS -----#
	#for file_neg in files_neg:
	tokens_neg = get_tokens_bigram(files_neg)
	i = 1
	while i < len(tokens_neg):
		bigram_list_neg.append([tokens_neg[i-1],tokens_neg[i]])
		i += 1

	i=0
	while i < len(bigram_list_neg):
		bigram = bigram_list_neg[i][0]+"|"+bigram_list_neg[i][1]
		if bigram not in bigramVocabulary:
			bigramVocabulary[bigram] = BigramItem(bigram)
			bigramVocabulary[bigram].word0 = bigram_list_neg[i][0]	
			bigramVocabulary[bigram].word1 = bigram_list_neg[i][1]
		
		bigramVocabulary[bigram].bigram_n_count += 1
		bigramVocabulary[bigram].bigram_all_count += 1	
		global tot_number_neg_bigram
		tot_number_neg_bigram += 1
		i += 1

	#---to print out the bigram ----#
	#for bigram in bigramVocabulary:
		#print("{"+bigramVocabulary[bigram].word0+"|"+bigramVocabulary[bigram].word1+"} p_count : "+str(bigramVocabulary[bigram].bigram_p_count)+ " n_count : "+str(bigramVocabulary[bigram].bigram_n_count)+ " all_count : "+str(bigramVocabulary[bigram].bigram_all_count))

	print("bigram length : "+str(len(bigramVocabulary)))

def CreateVocabulary():
	files_pos = glob.glob('./movies/train/P*.txt')
	files_neg = glob.glob('./movies/train/N*.txt')
	tokens_pos = get_tokens_bigram(files_pos)
	tokens_neg = get_tokens_bigram(files_neg)
	for token in tokens_pos:
		if token not in vocabulary:
			vocabulary[token] = VocItem(token)		

		vocabulary[token].p_count += 1
		global tot_number_pos
		tot_number_pos += 1
		vocabulary[token].all_count += 1

	for token in tokens_neg:
		if token not in vocabulary:
			vocabulary[token] = VocItem(token)		

		vocabulary[token].n_count += 1
		global tot_number_neg
		tot_number_neg += 1
		vocabulary[token].all_count += 1
	#print vocabulary
	#for key in vocabulary:
		#print(vocabulary[key].word +" - "+str(vocabulary[key].p_count)+" - "+str(vocabulary[key].n_count)+" - "+str(vocabulary[key].all_count)+"\n")
	for key in vocabulary:
		if vocabulary[key].all_count > 24 or key == "<s>":
			vocabulary_25plus[key] = vocabulary[key]


def smoothProbUnigram(w , rating , k=1):
	
	#if w in vocabulary_25plus:
	if rating == "p":
		smoothProb = (vocabulary_25plus[w].p_count + k) / (tot_number_pos + k* len(vocabulary_25plus))

	if rating == "n":
		smoothProb = (vocabulary_25plus[w].n_count + k) / (tot_number_neg + k* len(vocabulary_25plus))

	return smoothProb


def smoothProbBigram(wi_1 , wi , rating , k):

	bigramkey = wi_1+"|"+wi

	if rating == "p":
		smoothProb = (bigramVocabulary[bigramkey].bigram_p_count+k) / (vocabulary[wi_1].p_count + len(vocabulary_25plus)*k)
	if rating == "n":
		smoothProb = (bigramVocabulary[bigramkey].bigram_n_count+k) / (vocabulary[wi_1].n_count + len(vocabulary_25plus)*k)

	return smoothProb

# --------- PREDICT CLASS FOR EACH REVIEW IN TEST SET ----------------#
def ClassifyReviews(k):
	test_files = glob.glob('./movies/test/*.txt')
	#review_list = []
	
	
	#review_list = {}
	rating=""

	result =open("result-bigrams.txt","w+")
	for test_file in test_files:
		positive = 0
		negative = 0
		#get the file IDs
		filename = os.path.splitext(test_file)[0].split("/")[3]
		#review_list.append(filename)

		tokens = tokenize(glob.glob(test_file))
		# ----------- COUNT POSITIVE PROBABILITIES --- #
		# ----------- COUNT NEGATIVE PROBABILITIES --- #
		i = 1
		while i < len(tokens):
			bigramkey = tokens[i-1]+"|"+tokens[i]
			if bigramkey in bigramVocabulary and tokens[i-1] in vocabulary_25plus and tokens[i] in vocabulary_25plus :
				positive = positive + np.log(smoothProbBigram(tokens[i-1],tokens[i],"p",k))
				negative = negative + np.log(smoothProbBigram(tokens[i-1],tokens[i],"n",k))
			i += 1

	

		if positive > negative:
			rating =  "P"
		if positive < negative:
			rating =  "N"

		result.write(filename+"\t"+rating+"\n")


	result.close()

# --------- TESTING FUCNTIONS ----------------#


CreateVocabulary()

CreateBigramVocabulary()


ClassifyReviews(1)



#print("First elements of p_rows and n_rows")
#print(p_rows[0])
#print(n_rows[0])

#print("test wListProbLog NEGATIVE:  ")
#print(wListProbLog(["bad ","horrible "], "n", 1))

#print(wListProbLog(get_tokens(glob.glob('./movies/test/P-test25.txt')), "n", 1))

#print("test wListProbLog POSITIVE:  ")
#print(wListProbLog(get_tokens(glob.glob('./movies/test/P-test25.txt')), "p", 1))








