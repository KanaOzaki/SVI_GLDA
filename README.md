# SVI_GLDA
Gaussian LDA with Stochastic Variational Inference()

This Python code implements the Gaussian LDA with Stochastic Variational Inference.

Probabilistic topic models based on latent Dirichlet Allocation is widely used to extract latent topics from
document collections. In recent years, a number of extended topic models have been proposed, especially Gaussian
LDA(G-LDA) has attracted a lot of attention．G-LDA integrates topic modeling with word embeddings by replacing
discrete topic distribution over word types with multivariate Gaussian distribution on the word embedding space.
This can reflect semantic information into topics. In this paper, we use a G-LDA for our base topic model and
apply Stochastic Variational Inference (SVI), an efficient inference algorithm, to estimate topics. This encourages the model to analyze massive document collections, including those arriving in a stream.

# Installation

## Requirements
* python 3.6.6
* scipy
* numpy

## Files
* main.py ：　A main script of online VI package.
* Estimate.py ：　A package of functions for fitting G-LDA using stochastic optimization.
* printtopics.py ：　A Python script that displays the topics fit using the functions in Estimate.py.

python main.py -d 20ng -f ../data/corpus_20ng.txt -v ../data/vocab_20ng.txt —vec ../data/20ng_vectors_50d.txt -K 50 -t 1024 -k 1.0 -b 16     
/resultにiterationごとのパラメータ(gamma, mu, sigma)が保存される.
