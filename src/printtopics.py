# printtopics.py: Prints the words that are most prominent in a set of
# topics.


import sys, os, re, random, math, urllib2, time, cPickle
import numpy as np
import json as js
import optparse


def main():
    """
    Displays topics fit by GaussianLDA_SVI.py. The first column gives the
    (expected) most prominent words in the topics, the second column
    gives their (expected) relative prominence.
    """

    parser = optparse.OptionParser()
    parser.add_option("-d", dest="data", help="20ng or nips")
    parser.add_option("-v", dest="vocab", help="vocabrary filename")
    parser.add_option("--vec", dest="vector", help = "vector filename")
    parser.add_option("-K", dest="K", type="int", help="number of topics", default=20)
    parser.add_option("-t", dest = "tau0", type = float, help = "parameter tau0")
    parser.add_option("-k", dest = "kappa", type = float, help = "parameter kappa")
    parser.add_option("-b", dest = "S", type = int, help = "batch size")
    (options, args) = parser.parse_args()

    if(options.data == "20ng"):
        end =  18846/options.S - 1
    else:
        end = 1740/options.S -1

    def topic_assignment(vec, mu, sig):
        prob = np.zeros(options.K)
        for k in range(options.K):
            a = np.sqrt(np.linalg.det(sig[k]) * (2 * np.pi) ** sig[k].ndim)
            b = -0.5 * (vec - mu[k]).dot(np.linalg.inv(sig[k])).dot(vec - mu[k])
            prob[k] = np.exp(b)/a

        return prob

    path = '../result/' + options.data + '/K' + str(options.K) +  '/S'+ str(options.S) + '/tau' + str(options.tau0) + '_kappa' + str(options.kappa) + '/'

    vocab = str.split(file(options.vocab).read())
    fr = open(options.vector, 'r')
    word_vecs = js.load(fr)
    fr.close()
    mu = np.load(path + 'mu/mu-' + str(end) + ".npy")
    sigma = np.load(path + 'sigma/sigma-' + str(end) + ".npy")

    #open result_file
    result_file = open(path + 'topic_' + str(options.S) + '_' + str(options.tau0) + '_' + str(options.kappa) + '.txt', 'w')
    rank = [[] for k in range(options.K)]
    prob_list = [[] for k in range(options.K)]
    for word_id in word_vecs:
        prob = topic_assignment(word_vecs[word_id], mu, sigma)
        for k in range(0, options.K):
            rank[k].append([word_id, prob[k]])
            prob_list[k].append(prob[k])

    for k in range(0, options.K):
        sum_prob = sum(prob_list[k])
        for m in range(len(rank[k])):
            rank[k][m][1] = rank[k][m][1]/sum_prob
        rank[k] = sorted(rank[k], key=lambda x:x[1], reverse=True)
        result_file.write('topic ' + str(k) + "\n" )
        for l in range(0, 10):
            result_file.write('%20s  \t---\t  %.16f' % (vocab[int(rank[k][l][0])], rank[k][l][1]))
            result_file.write("\n")

    result_file.close()

if __name__ == '__main__':
    main()
