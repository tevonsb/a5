#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    def __init__(self, input_size, dropout_rate):
        super(Highway, self).__init__()
        self.proj = nn.Linear(input_size, input_size, bias=True)
        self.gate = nn.Linear(input_size, input_size, bias=True)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)


    def forward(self, x):
        x_proj = F.relu(self.gate(x))
        x_gate = torch.sigmoid(self.proj(x))

        x_highway = x_gate * x_proj + (1-x_gate) * x
        w_word_embed =  self.dropout(x_highway)
        return w_word_embed
### END YOUR CODE
