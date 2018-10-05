# -*- coding:utf-8 -*-
import pickle
import numpy as np
import tensorflow as tf
import collections
from tqdm import tqdm


def skip_gram_demo():
    with open('/home/tf/dl_fun/13_word2Vec/data/wiki.zh.word.text', 'rb') as fr:
        lines = fr.readlines()
    lines = [line.decode('utf-8') for line in lines]
    words = ' '.join(lines)
    words = words.replace('\n', '').split(' ')
    print('共%d个词' % len(words))

if __name__ == '__main__':
    skip_gram_demo()
