#!/usr/bin/python

import numpy as np
import pandas as pd
import sys, getopt
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import math
import os.path
import argparse


class EmbeddingFeaturesGenerator:
	'''
	Class that generates embedding representation of sentences. 
	'''

	#name of the default embedding file. 
	DEFAULT_EMBEDDINGS_FILE = "./wiki_embeddings.txt"

	#various schemes of combining sentence word embeddings
	TYPE_1="min_max"
	TYPE_2="min"
	TYPE_3="max"
	TYPE_4="average"

	def __init__(self,embeddings_file=DEFAULT_EMBEDDINGS_FILE,type=TYPE_1,verbose=1):
		self._verbose=verbose
		self._type=type;
		self.load_embeddings(embeddings_file)

	def load_embeddings(self,embeddings_file):
		'''
		Method to load a txt file containing the word embeddings. The format for each line should be
    	[word embedding vector] 
    	'''

		if self._verbose:
			print " Reading word embeddings...."
		
		embd_df=pd.read_csv(embeddings_file, header=None, delimiter=r"\s+", index_col = False)
		embedding_size=len(embd_df.columns)-1;
		cols=["embd_"+str(i) for i in range(embedding_size+1)]
		cols[0]="word";	
		embd_df.columns= cols;
		self.__embedding_size=embedding_size;
		self.__embeddings=embd_df

	def get_embeddings_vector(self,word):
		'''
		Method to retrieve a work embedding vector for a specific word  
    	'''
		return self.__embeddings.loc[self.__embeddings['word'] == word, self.__embeddings.dtypes == float].as_matrix();

	def get_embeddings_representation(self,sentence):
		'''
		Method that returns a word embedding representation (i.e. features) for the input sentence.
    	'''
		words=sentence.split();	

		arr = np.empty(shape=[0, self.__embedding_size])
	
		#put all embeddings for this sentence in an array
		for word in words:
			arr=np.append(arr,self.get_embeddings_vector(word),axis=0)
			
		if arr.size==0:
			arr=np.zeros((2,self.__embedding_size));

		#estimate the embedding representation of the sentence, depending on the selected scheme
		if self._type==self.TYPE_1:
			min_embedding=np.amin(arr,axis=0)
			max_embedding=np.amax(arr,axis=0)
			feature=np.concatenate((min_embedding,max_embedding),axis=0)
		elif self._type==self.TYPE_2:
			feature=np.amin(arr,axis=0)
		elif self._type==self.TYPE_3:
			feature=np.amax(arr,axis=0)
		else:
			feature=np.average(arr,axis=0)

		return feature

	def extract_features(self,filename):
		'''
		Method that generates the word embedding representation (i.e. features) for the input file.
    	'''

		if self._verbose:
			print " Extracting embedding representation for",filename,"..."

		features_size=0;
		if self._type==self.TYPE_1:
			features_size=2*self.__embedding_size
		else:
			features_size=self.__embedding_size


		X=np.empty(shape=[0, features_size])
		Y=[]

		f = open(filename)
		line = f.readline()
		cnt=0

		while line:
			parts=line.split('\t');
			label=parts[0];
			sentence=parts[1];
		
			feature=self.get_embeddings_representation(sentence);

			#append to X,Y
			X=np.append(X,np.reshape(feature,(1,features_size)),axis=0);
			Y.append(label);


			if self._verbose:
				#get number of lines
				num_lines = sum(1 for line in open(filename))
				percentage=cnt/(num_lines*1.0)
				sys.stdout.write('\r')
				sys.stdout.write(" [%-50s] %d%%" % ('='*int(percentage*50), percentage*100))
				sys.stdout.flush()

			cnt=cnt+1;
			line = f.readline()
		f.close()

		if self._verbose:
			sys.stdout.write("\n");

		return X,Y


def main(train_set,test_set,embeddings_file=EmbeddingFeaturesGenerator.DEFAULT_EMBEDDINGS_FILE):
	
	if not os.path.exists(embeddings_file):
		if embeddings_file==EmbeddingFeaturesGenerator.DEFAULT_EMBEDDINGS_FILE:
			print "---> Embeddings file not available, downloading default pre-trained embeddings...."
			os.system('wget https://www.dropbox.com/s/h87tstu4awtvgew/wiki_embeddings.txt')
		else: 
			print "Embeddings file not found."
			exit(1)

	print ">Feature Extraction....";

	feature_gen=EmbeddingFeaturesGenerator(embeddings_file,EmbeddingFeaturesGenerator.TYPE_1);

	X_train,Y_train=feature_gen.extract_features(train_set)
	X_test,Y_test=feature_gen.extract_features(test_set)

	print ">Training SVM Classifier....";
	classifier = SVC()
	classifier.fit(X_train, np.asarray(Y_train))
	
	print ">Predicting.....";
	predicted=classifier.predict(X_test)

	print ">Classification Results:";
	print(classification_report(Y_test, predicted))
	print(accuracy_score(Y_test, predicted))
	print(confusion_matrix(Y_test, predicted))


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Text-based classification using word-embedding representation')
	parser.add_argument('--train_set', metavar='path', required=True, help='File containing sentences to be used for training')
	parser.add_argument('--test_set', metavar='path', required=True, help='File containing sentences to be used for testing')
	parser.add_argument('--embeddings_file', metavar='path', required=False, help='Text file containing pre-trained word embeddings')
	args = parser.parse_args()
	
	main(train_set=args.train_set,test_set=args.test_set,embeddings_file=args.embeddings_file)