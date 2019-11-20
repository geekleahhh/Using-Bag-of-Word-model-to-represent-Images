import numpy as np
inverted_index=np.load('inverted_index.npy')
word=input("please enter your word(0-99)\n")
word=int(word)
print(word)
print(inverted_index[word])