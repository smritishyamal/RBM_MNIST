# RBM_MNIST
RBM for MNIST data

A program that adjusts the parameters of an restricted Boltzman machine (RBM) to minimize the KL divergence between the empirical distribution of the MNIST dataset of handwritten digits and the mariginal distribution of the RBM. 

The elements of the MNIST dataset are defined with integer values, in contrast to the visible variables of an RBM, which are binary. We treat each element of the MNIST dataset as a product of Bernoulli distributions, with the probability of each pixel being 1 proportional to its value in the unprocessed MNIST dataset.
