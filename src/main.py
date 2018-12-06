#!/usr/bin/python
import cPickle, string, getopt, sys, random, time, re, pprint
from numpy.random import *
import numpy as np
import pickle as pk
import json as js
import os
import optparse

import Estimate as GLDA
import data

"""
In this implementaition, we use GaussianLDA model with SVI.
"""


def main():

        parser = optparse.OptionParser()
        parser.add_option("-d", dest = "data", help = "20ng or nips")
	parser.add_option("-f", dest="filename", help="corpus filename")
	parser.add_option("-v", dest="vocab", help="vocabrary filename")
	parser.add_option("--vec", dest="vector", help = "vector filename")
	parser.add_option("-t", dest = "tau0", type = float, help = "parameter tau0")
	parser.add_option("-k", dest = "kappa", type = float, help = "parameter kappa")
	parser.add_option("-b", dest = "S", type = int, help = "batch size")
	parser.add_option("-K", dest="K", type="int", help="number of topics", default=20)
	(options, args) = parser.parse_args()

	# alpha : hyper parameter of the dirichlet distribution 1/K
	alpha = float(1)/options.K

	path = '../result/'+ options.data + '/K' + str(options.K) + '/S'+ str(options.S) + '/tau' + str(options.tau0) + '_kappa' + str(options.kappa) + '/'
	#os.makedirs(path + 'gamma/')
	os.makedirs(path + 'mu/')
	#os.makedirs(path + 'sigma/')
	#os.makedirs(path + 'E_gamma/')
	#os.makedirs(path + 'E_gauss/')


	Data = data.PrepareData(options.vocab, options.filename, options.vector)

	Data.make_corpus_vector()
	corpus_data = Data.all_corpus
	word_vecs_doc = Data.word_vecs_doc
	word_vecs_all = Data.word_vecs_all
	corpus_vocab = Data.vocab_in_corpus
	Doc_num = len(corpus_data)

        # make a word_vecs data only for the first time
	if(False):
                path2 = '../word_vecs/'
                #os.makedirs(path2)
                fw = open(path2 + 'word_vecs_nips.json', 'w')
                js.dump(word_vecs_all, fw)
                fw.close()


	batch_num = Doc_num/options.S #1177
	print (batch_num)

	olda = GLDA.SVILDA(corpus_vocab, word_vecs_all, options.K, Data.D, alpha, options.tau0, options.kappa)

	start = time.time()

	for iteration in range(batch_num):
		docset = [corpus_data[i] for i in range(iteration*options.S, (iteration + 1) * options.S)]
		vecset = word_vecs_doc[iteration*options.S : ((iteration + 1) * options.S)]

		if(iteration == batch_num-1):
			docset = [corpus_data[i] for i in range(iteration*options.S, Doc_num - 1)]
			vecset = word_vecs_doc[iteration*options.S : (Doc_num - 1)]

		# Give them to online LDA
		(gamma, bound) = olda.update_zeta_docs(docset, vecset)

		# Compute an estimate of held_out perplexity
		(wordids, wordcts) = olda.count_words(docset)
		perwordbound = bound * len(docset) / (Doc_num * sum(map(sum, wordcts)))
		if(iteration %10 == 0):
			#sigma = (olda._psi.transpose() / (olda._nu - Data.D + 1)).transpose()
			np.save(path + 'mu/' +'mu-%d.npy' % iteration, olda._m)
			#np.save(path + 'sigma/' +'sigma-%d.npy' % iteration, sigma)
			#np.save(path + 'gamma/' +'gamma-%d.npy' % iteration, gamma)
			#np.save(path + 'E_gamma/' + 'E_gamma-%d.npy' % iteration, olda._expEloggauss)
			#np.save(path + 'E_gauss/' + 'E_gauss-%d.npy' % iteration, olda._expEloggauss)
			elapsed_time = time.time() - start
			print ('iteration : %d' % iteration)
			print ("elapsed_time:{0}".format(elapsed_time)) + "[sec]"
			print ("perplexity : {}".format(np.exp(-perwordbound)))
			print ("-"*30)

if __name__ == '__main__':
	main()
