#!/bin/bash

echo "--> Word Embeddings Test <--"
python word_embedding_features.py --train_set ./smsspamcollection/train.txt --test_set ./smsspamcollection/test.txt --embeddings_file ./wiki_embeddings.txt

