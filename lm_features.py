#!/usr/bin/python

import numpy as np
import pandas as pd



class EmbeddingFeaturesGenerator:

	def train(self,filename):
		f = open(filename)
		line = f.readline()
		cnt=0

		while line:
			parts=line.split('\t');
			label=parts[0];
			sentence=parts[1];
			print sentence;

			#words=sentence.split();	





			line = f.readline()
		f.close()		

def main(train_set,test_set):
	
	lm=EmbeddingFeaturesGenerator()
	lm.train(train_set)

if __name__ == "__main__":
	main("./smsspamcollection/train.txt","smsspamcollection/test.txt")