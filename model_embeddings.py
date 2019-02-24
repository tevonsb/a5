#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        pad_token_idx = vocab['<pad>']
        # self.embeddings = nn.Embedding(len(vocab), embed_size, padding_idx=pad_token_idx)
        self.e_char = 50
        self.embeddings = nn.Embedding(len(vocab.id2char), self.e_char, padding_idx=pad_token_idx)
        self.vocab = vocab
        self.embed_size = embed_size
        self.cnn = CNN(self.e_char, self.embed_size)
        self.highway = Highway(self.embed_size, 0.3) # May need to change dropout
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        # print(input.shape)
        batch_size = input.shape[1]
        embedding = self.embeddings(input)
        # print("Embedded Shape")
        # print(embedding.shape)
        embedding = embedding.view((embedding.shape[0]*embedding.shape[1], embedding.shape[2], embedding.shape[3]))

        embedding = embedding.permute(0, 2, 1)
        # print("Embedded Shape")
        # print(embedding.shape)
        convoluted = self.cnn.forward(embedding)
        # print("Convoluted Shape")
        # print(convoluted.shape)
        embedded = self.highway.forward(convoluted)
        # print('after highway')
        # print(embedded.shape)
        embedded = embedded.view((int(embedded.shape[0]/batch_size), batch_size, embedded.shape[1]))

        return embedded


        ### END YOUR CODE
