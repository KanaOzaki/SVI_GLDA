#!/usr/bin/python
import cPickle, string, getopt, sys, random, time, re, pprint
from numpy.random import *
import numpy as np
import pickle as pk
import json as js
import os

np.random.seed(100000001)

"""
Prepare corpus, vector.
"""

class PrepareData(object):

    def __init__(self, vocab_file, corpus_file, vector_file):
		self.vocab_file = vocab_file
		self.corpus_file = corpus_file
		self.vector_file = vector_file
		self.D = 50
		self.vocab_in_corpus = set([])

    def make_corpus_vector(self):
		self.make_vocab()
		self.make_corpus()
		self.make_vector()

    def	make_vocab(self):
		"""
		Make vocabrary dict
		"""
		vocab_data = file(self.vocab_file).readlines()
		self.vocab = dict()
		for word in vocab_data:
			word = word.lower()
			word = re.sub(r'\W', '', word)
			self.vocab[len(self.vocab)] = word

		return self.vocab

    def make_corpus(self):
		"""
		Make corpus list
		"""
		corpus_data = file(self.corpus_file).readlines()
		self.all_corpus = {}
		for index, doc in enumerate(corpus_data):
			tmp = doc.split(' ')
			tmp.pop()
			corpus = []
			for word_id in tmp:
				corpus += [int(word_id)]
				self.vocab_in_corpus.add(int(word_id))
			self.all_corpus[index] = corpus

    def make_vector(self):
		vectors = file(self.vector_file).readlines()
		word_vecs = {}
		for id_vec in vectors:
			vec_list = id_vec.split(" ")
			word = vec_list[0]
			vec_list.pop(0)
			for i in range(len(vec_list)):
				vec_list[i] = float(vec_list[i])
			word_vecs[word] = vec_list

		self.word_vecs_all = {}
		self.word_vecs_doc = []

		for word_ids in self.all_corpus.values():
			tmp = {}
			#word_ids_copy = word_ids
			#print self.all_corpus
			for word_id in word_ids:
				word = self.vocab[word_id]
				try:
					self.word_vecs_all[word_id] = word_vecs[word]
					tmp[word_id] = word_vecs[word]
				except:
					#word_ids_copy.remove(word_id)
					if(word_id in self.vocab_in_corpus):
						self.vocab_in_corpus.remove(word_id)
			#word_ids = word_ids_copy
			self.word_vecs_doc.append(tmp)
