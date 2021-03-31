Code used in:

Barry M. Dillon, Christof Sauer, Peter Sorrenson, Tilman Plehn

In this repo we build a mixture model VAE to analyse jet images.
For physics applications this model could have advantages over the more widely used Gaussian latent space, for unsupervised jet image classification this seems to be the case.
The Dirichlet latent space has more structure and seems to accommodate different sub-classes within the dataset nicely.

This is inspired by the Latent Dirichlet Allocation topic-models, but seems to be more versatile and more powerful.

The VAE has a Dirichlet latent space, where we use a softmax Gaussian approximation to the Dirichlet distribution.
An N-dimensional Dirichlet is defined on an N-dimensional simplex, and in the softmax Gaussian approximation has 2xN parameters (N means and N variances) which are related to the Dirichlet hyper-parameters by a simple relation.

The neural network architecture is very simple:

Encoder:
 - Input layer for a pixelised image
 - One hidden layer with 100 nodes and SeLU activations
 - 2xN linear outputs for an N-dimensional latent space

Sampling:
 - Sample N values from the N-dimensional Gaussian with the usual re-parameterisation step
 - Pass the vector through a softmax

Decoder:
 - N nodes on the input layer
 - No hidden layers and no biases
 - Output layer has the same dimension as the input layer and a softmax activation

The loss function can be derived from the ELBO of a mixture model, but consists of the standard two terms:
 - reconstruction loss: cross-entropy
 - latent loss: KL divergence between Dirichlet prior and posterior

The weights of the decoder can be directly interpreted as mixtures learned by the VAE.
Sort of a continuous generalisation of "topics".
With one-hot encoded inputs this is similar to LDA.

Refs:
 - AutoEncoding VI for topic models (https://arxiv.org/pdf/1703.01488.pdf)
 - Dirichlet VAE (https://arxiv.org/pdf/1901.02739.pdf)
 - LDA (https://jmlr.org/papers/volume3/blei03a/blei03a.pdf)
 - LDA for jets I (https://arxiv.org/abs/1904.04200)
 - LDA for jets II (https://arxiv.org/abs/2005.12319)
 - LDA for jets github repo (https://github.com/bmdillon/lda-jet-substructure)
 - VAEs for jets (https://arxiv.org/abs/1808.08979 & https://arxiv.org/abs/1808.08992 & https://arxiv.org/abs/2007.01850)
