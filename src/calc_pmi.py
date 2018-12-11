#!/usr/bin/python
import sys
import time
from numpy.random import *
import numpy as n
import pickle as pk
import json as js

DIR_PATH_WIKI = './wiki_data/'

def calc_pmi(word_x, word_y, wids, bow):

	
	# number of words in the data
	N = 701599094
	# N = 23

	# get N_x, N_y from wids
	# if there are no much words, return 0
	try:
		N_x = wids[word_x][1]		
		N_y = wids[word_y][1]
	except:
		return 0	

	# get N_xy from bow
	N_xy = 0
	for freq_d in bow:
		M_x = freq_d.get(wids[word_x][0])
		M_y = freq_d.get(wids[word_y][0])

		if(min(M_x, M_y) != None):
			N_xy += min(M_x, M_y)			

	# case : N_xy = 0 
	# word_x and word_y never co_occred in the same document.		
	if(N_xy == 0):
		return 0

	PMI = n.log2(N_xy * N / float(N_x * N_y))	

	return PMI

def main():

	start=time.time()

	argvs = sys.argv
	# file_name
	filename = argvs[1]

	### make word_topics (K * 10 list)
	data = open(filename, 'r')
	word_topics = []
	for line in data:
		words = line.split()
		if(words[0] == 'topic'):	
			if(words[1] != '0'):
				word_topics.append(word_topic)
			word_topic = []	
		else:
			word = words[0]
			word_topic.append(word)
	word_topics.append(word_topic)		
	data.close()		

	for j in range(0, len(word_topics)):
		print ('topic : %d' % j)
		print (word_topics[j])

	K = len(word_topics)
	#print K 

	### load wiki_data
	with open(DIR_PATH_WIKI + 'word_id_freq.pickle', 'rb') as f1:
		wids = pk.load(f1)
	with open(DIR_PATH_WIKI + 'bow_list.pickle', 'rb') as f2:
		bow = pk.load(f2)	

	PMI_topics_mean = []
        PMI_topics_sum = []
	for k in range(0,K):
		PMIs = []
		for i in range(0,10):
			for j in range(i+1,10):
				PMI = calc_pmi(word_topics[k][i], word_topics[k][j], wids, bow)
				PMIs.append(PMI)
		PMI_mean = sum(PMIs)/45.
                PMI_sum = sum(PMIs)
                PMI_topics_mean.append(PMI_mean)
                PMI_topics_sum.append(PMI_sum)

    
	for i in range(0, K):
                print ('topic %d    mean : %.16f sum : %.16f' % (i, PMI_topics_mean[i], PMI_topics_sum[i]))
	
	all_PMI_mean = sum(PMI_topics_mean)/float(K)
	print ('All_PMI_mean : %.16f' % all_PMI_mean)            
	

	elapsed_time = time.time() - start
	print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"

if __name__ == '__main__':
	main()

