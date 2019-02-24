#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
class CNN(nn.Module):
    def __init__(self, e_char, e_word):
        super(CNN, self).__init__()
        # kernel size of 5
        self.kernel_size = 5
        self.conv = nn.Conv1d(e_char, e_word, self.kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.MaxPool = nn.MaxPool1d(21)

    def forward(self, x):
        x_conv = self.conv(x)

        relud = F.relu(x_conv)
        x_conv_out = self.MaxPool(relud)
        x_conv_out = x_conv_out.squeeze(dim=2)
        # x_conv_out = torch.max(relud, 2)[0]
        # print("maxpooled size")
        # print(x_conv_out.shape)
        return x_conv_out
### END YOUR CODE
