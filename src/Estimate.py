#!/usr/bin/python
import sys, re, time, string, math
import numpy as n
from scipy.special import gammaln, psi
import random

n.random.seed(100000001)
meanchangethresh = 0.001

def dirichlet_expectation(alpha):
	"""
	For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
	"""

	if (len(alpha.shape) == 1):
		return(psi(alpha) - psi(n.sum(alpha)))
	return (psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis])




class SVILDA:
	"""
	Implements SVI for Gaussian LDA.
	"""

	def __init__(self, vocab, word_vecs, K, D, alpha, tau0, kappa):
		"""
		Arguments :
		K : Number of topic
		vocab : A set of words to recognize. When analysing documents, any word
				not in this set will be ignored.
		D : Total number of documents in the population. For a fixed corpus,
		    this is the size of the corpus. In the truly online setting, this
		    can be an estimate of the maximum number of documents that could
		    ever be seen.
		alpha : Hyperparameter for prior on weight vectors theta
		tau0 : A (positive) learning parameter that downweights early itrations
		kappa : Learning rate: exponential decay rate---should be between
				(0.5, 1.0] to guarantee asympotic convergence.

		Note that if you pass the same set of D documents in every time and
		kappa = 0 this class can also be used to batch VB.

		"""

		self._vocab = vocab
		self._wordvecs = word_vecs
		self._K = K
		self._W = max(vocab)+1
		self._D = D
		self._alpha = alpha
		self._tau0 = tau0
		self._kappa = kappa
		self._updatect = 0
		self._M = 50

		# Initialize the variational distribution q(mu, sigma|zeta) = NIW(mu, sigma|zeta)
		wordvecs = n.vstack(self._wordvecs.values())#n.vstack([self._wordvecs[i].values() for i in range(0, self._D)])
		self._m0 = []
		for k in range(0, self._K):
			num = random.randrange(len(wordvecs))
			self._m0.append(list(wordvecs[num]))
		self._m0 = n.array(self._m0)
		self._beta0 = n.array([0.01] * self._K)
		#self._psi0 = n.array([n.sum(wordvecs[w].reshape(self._M,1).dot(wordvecs[w].reshape(1,self._M)) for w in range(len(wordvecs)))] * self._K)
                self._psi0 = n.array([3*self._M * n.identity(self._M)]*self._K)
		self._nu0 = n.array([wordvecs.shape[1]] * self._K)
		self._m = self._m0
		self._beta = self._beta0
		self._psi = self._psi0
		self._nu = self._nu0
		self._Eloggauss = 1*n.random.gamma(100., 1./100, (self._K, self._W)) 
		self._expEloggauss = n.exp(self._Eloggauss)
		self._Eloggauss0 = self._Eloggauss

	def count_words(self,docs):
		"""
		Parse a set of documents into two lists of lists of word ids and counts.

		Returns a pair of lists of lists.

		The first, wordids, says that vocablary tokens are present in document.
		wordids[i][j] gives the jth unique token present in documents i.
		(Don't count on these tokens being in any paricular order.)

		The second, wordcts, says how many times each vocablary token is preset.
		wordcts[i][j] is the number of times that the tokens given by word[i][j]
		appers in document i.
		"""

		D = len(docs)

		wordids = list()
		wordcts = list()
		for d in range(0,D):
			words = docs[d]
			ddict = dict()
			for word in words:
				if(word in self._vocab):
					if (not word in ddict):
						ddict[word] = 0
					ddict[word] += 1
			wordids.append(ddict.keys())
			wordcts.append(ddict.values())

		return 	((wordids, wordcts))


	def do_e_step(self, wordids, wordcts):
		batchD = len(wordids)

		# Initialize the variational distribution q(theta|gamma) for the mini-batch.
		gamma = 1*n.random.gamma(100., 1./100, (batchD, self._K))
		Elogtheta = dirichlet_expectation(gamma)
		expElogtheta = n.exp(Elogtheta)

		sstats = n.zeros(self._Eloggauss.shape)
		# Now, for each document d update that document's gamma and phi.
		it = 0
		meanchange = 0
		for d in range(0, batchD):
			# These are mostly just shorthand (but might help cache licality)
			ids = wordids[d]
			cts = wordcts[d]
			gammad = gamma[d,:]
			Elogthetad = Elogtheta[d,:]
			#print Elogthetad
			expElogthetad = expElogtheta[d,:]
			#print expElogthetad
			expEloggaussd = self._expEloggauss[:,ids]
			# The optimal phi_{dwk} is propotional to expElogthetad_k * expEloggaussd_w.
			# phinom is the normalizer.
			phinorm = n.dot(expElogthetad, expEloggaussd) + 1e-100
			# Iterate between gamma and phi until convergence
			for it in range(0, 100):
				lastgamma = gammad
				# We represent phi implicitly to save memory and time.
				# Substituting the values of the optimal phi back into
				# the update for gamma gives this update. Cf. Lee&Seung 2001.
				gammad = self._alpha + expElogthetad * n.dot(cts / phinorm, expEloggaussd.T)
				Elogthetad = dirichlet_expectation(gammad)
				expElogthetad = n.exp(Elogthetad)
				phinorm = n.dot(expElogthetad, expEloggaussd) + 1e-100
				# If gamma hasn't changed much, we're done.
				meanchange = n.mean(abs(gammad - lastgamma))
				if (meanchange < meanchangethresh):
					break
			gamma[d,:] = gammad
			# Contribution of document d to the expected suffient statistict for the M setp.
			sstats[:, ids] += n.outer(expElogthetad.T, cts / phinorm)

		# This step finishes computing the sufficient statistics for the M step, so that
		# sstats[k, w] = \sum_d n_{dw} * phi_{dwk} = \sum_d n_{dw} * exp{Elogtheta_{dk} + Eloggauss_{kw}} / phinorm_{dw} .
		#print sstats.shape
		#print self._Eloggauss
		sstats = sstats * self._expEloggauss

		return ((gamma, sstats))

	def do_e_step_docs(self, docs):
		"""
		Given a mini-batch of documents, estimates the parameters gammma
		controlling the bariational distribution over the topic weights
		for each document in the mini-batch.

		Arguments :
		docs : List of D documents. Each document must be represented as word ids.
			   (Word order is unimportant.) Any words not in the vocablary will
			   be ignored.

		Returns a tuple containing the estimated values of gamma,
		as well as suggicient statistics needed to update zeta.
		"""

		(wordids, wordcts) = self.count_words(docs)

		return self.do_e_step(wordids, wordcts)


	def calc_Eloggauss(self, m, beta, Psi, nu, wordvecs, batchD):
		"""
		For a vector mu_k, sigma_k ~ NIW(zeta), computes E[log(v|mu, sigma)] given zeta.
		"""

		Eloggauss = n.zeros((self._K , self._W))

		wordvec_d = n.array(wordvecs.values())
		W_d = len(wordvec_d)

		Eloggauss_d = n.zeros((self._K, W_d))

		for k in range(self._K):
			# a1 : <sigma_k^-1>  = nu_k * Psi_k^-1
			a1 = nu[k] * n.linalg.inv(Psi[k])
			# a2 : <sigma_k^-1*mu_k> = nu_k * Psi_k^-1 * m_k
			a2 = nu[k] * n.linalg.inv(Psi[k]).dot(m[k])
			# a3 : <mu_k.T*sigma_k^-1*mu_k> = nu_k * m_k.T * Psi_k^-1 * m_k + batchD/beta_k
			a3 = nu[k] * (m[k].T.dot(n.linalg.inv(Psi[k]))).dot(m[k]) + float(batchD)/beta[k]
			# a4 : <log|sigma_k^-1|> = sum_d psi((nu_k + 1 - d)/ 2) + batchD*log2 + log|Psi_k^-1|
			#a4 = batchD * math.log(2) + math.log(n.linalg.det(n.linalg.inv(Psi[k])))
			a4 = batchD * math.log(2) + math.log(1./n.linalg.det(Psi[k]))
			for d in range(1, batchD+1):
				a4 += psi((nu[k] + 1 - d)/2)
			for w in range(W_d):
				# Eloggauss[k][w] = -1/2*v_{dw}.T*<sigma_k^-1>*v_{dw} + v_{dw}.T*<sigma_k^-1*mu_k>
				#                   - 1/2*<mu_k.T*sigma_k^-1*mu_k> - 1/2*<log|sigma_k^-1|>
				Eloggauss_d[k][w] =  - 0.5 * (wordvec_d[w].T.dot(a1)).dot(wordvec_d[w]) + wordvec_d[w].T.dot(a2) - 0.5 * a3  + 0.5 * a4

		Eloggauss[:, wordvecs.keys()] = Eloggauss_d

		return Eloggauss



	def update_zeta_docs(self, docs, vecs):
		"""
		First does an E step on the mini-batch given in wordids and wordcts,
		then uses the result of that E step to update the variational parameter
		matrix zeta (zeta = (m, beta, Psi, nu)).

		Arguments :
		docs : List of D documents. Each document must be represented as word ids.
			   (Word order is unimportant.) Any words not in the vocablary will
			   be ignored.

		Returns gamma, the parameters to the variational distribution over the topic
		weights theta for the documents analyzed in this update.

		Also returns an estimate of the variational bound for the entire corpus for
		the OLD setting of zeta based on the documents passed in. This can be used
		as a (possibly very noisy) estimate of held_out likelihood.
		"""

		# rhot will be between 0 and 1, and says how mach to weight
		# the information we got from this mini-batch.
		rhot = pow(self._tau0 + self._updatect, -self._kappa)
		self._rhot = rhot
		# Do an E step to update gamma, phi | zeta for this mini-batch.
		# This is also returns the information about phi that we need to update zeta.
		#docs_id = [self._wordvecs[i].keys() for i in range(0, len(docs))]
		(gamma, sstats) = self.do_e_step_docs(docs)
		# Estimate held-out likelihood for current values of zeta.
		# bound = self.approx_bound_docs(docs_id, gamma)

		# Do an M step to update zeta
		# make wordvecs matrix : 1 * W
		wordvecs = {}
		for vec_d in vecs:
			for wid in vec_d:
				if( wid in self._vocab):
					wordvecs[wid] = vec_d[wid]

		# To compute zeta, we only use the words in docs
		vec_matrix_d = n.array(wordvecs.values())
		W_d = len(vec_matrix_d)

		# Update zeta based on documents.

		# sstats_d[i]
		sstats_d = sstats[ : , wordvecs.keys()]

		# sstats_sum[k] = sigma_w n_{tw} * phi_{twk}
		sstats_sum = n.array([n.sum(sstats_d[k]) for k in range(self._K)])
		beta = self._beta0 + float(self._D) * sstats_sum / len(docs)
		nu = self._nu0 + float(self._D) * sstats_sum / len(docs)

		scaled_topic_mean = ((sstats_d.dot(vec_matrix_d).transpose()) / sstats_sum).transpose()

		mean_centored = n.array([vec_matrix_d] * self._K) - n.array([scaled_topic_mean] * W_d).transpose((1,0,2))

		scaled_topic_cov = float(self._D) / len(docs) * n.array([[n.sum(sstats_d[k][w] * mean_centored[k][w].reshape(self._M,1).dot(mean_centored[k][w].reshape(1,self._M)) for w in range(W_d))] for k in range(self._K)])

		scaled_topic_cov = scaled_topic_cov.reshape(self._K,self._M,self._M)

		vk_mu = scaled_topic_mean - self._m

		m = (self._beta0[0] * self._m + (self._D * scaled_topic_mean.transpose() * sstats_sum).transpose()).transpose() /  (beta * len(docs))
		m = m.transpose()

		t = n.array([vk_mu[k].reshape(self._M,1).dot(vk_mu[k].reshape(1,self._M)) for k in range(self._K)])

		psi = self._psi0[0] + scaled_topic_cov + n.array([(self._beta0[0] * float(self._D) /len(docs) * sstats_sum)[k] * t[k]/beta[k] for k in range(self._K)])

		self._m = (1-rhot) * self._m + rhot * m
		self._beta = (1-rhot) * self._beta + rhot * beta
		self._psi = (1-rhot) * self._psi + rhot * psi
		self._nu = (1-rhot) * self._nu + rhot * nu

		# calc expectation
		self._Eloggauss = self.calc_Eloggauss(self._m, self._beta, self._psi, self._nu, wordvecs, len(docs))
		self._expEloggauss = n.exp(self._Eloggauss)
		self._updatect += 1

		return gamma
