#-*- coding: UTF-8 -*-
#  Author: Angela Chapman
#  Date: 8/6/2014
#
#  This file contains code to accompany the Kaggle tutorial
#  "Deep learning goes to the movies".  The code in this file
#  is for Parts 2 and 3 of the tutorial, which cover how to
#  train a model using Word2Vec.
#
# *************************************** #


# ****** Read the two training sets and the test set
#
import pandas as pd
import os
from nltk.corpus import stopwords
import nltk.data
import logging
import numpy as np  # Make sure that numpy is imported
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.append("..")
from KaggleWord2VecUtility import KaggleWord2VecUtility


# ****** Define functions to create average word vectors
#

def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    #对于每个评论中的每个word，找到它在word中的vector，并将所有word的vector相加然后取平均   vecotr是300维
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    #返回的300维的vector是一个review的特征
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print "Review %d of %d" % (counter, len(reviews))
       #
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, \
           num_features)
       #
       # Increment the counter
       counter = counter + 1.
    #这里返回的review是数据集每条review的矩阵，每一行是一个review，一共有len(reviews)列
    return reviewFeatureVecs


def getCleanReviews(reviews):
    clean_reviews = []
    for review in reviews["review"]:
        clean_reviews.append( KaggleWord2VecUtility.review_to_wordlist( review, remove_stopwords=True ))
    return clean_reviews



if __name__ == '__main__':

    # Read data from files
    train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, delimiter="\t", quoting=3 )
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", quoting=3 )
    unlabeled_train = pd.read_csv( os.path.join(os.path.dirname(__file__), 'data', "unlabeledTrainData.tsv"), header=0,  delimiter="\t", quoting=3 )

    # Verify the number of reviews that were read (100,000 in total)
    print "Read %d labeled train reviews, %d labeled test reviews, " \
     "and %d unlabeled reviews\n" % (train["review"].size,
     test["review"].size, unlabeled_train["review"].size )

    num_features = 300    # Word vector dimensionality

    model = Word2Vec.load("300features_40minwords_10context")
 
    # ****** Create average vectors for the training and test sets
    #
    print "Creating average feature vecs for training reviews"

    trainDataVecs = getAvgFeatureVecs( getCleanReviews(train), model, num_features )
    
    print "Creating average feature vecs for test reviews"

    testDataVecs = getAvgFeatureVecs( getCleanReviews(test), model, num_features )

    
    # ****** Fit a random forest to the training set, then make predictions
    #
    # Fit a random forest to the training data, using 100 trees
    forest = RandomForestClassifier( n_estimators = 100 )

    print "Fitting a random forest to labeled training data..."
    forest = forest.fit( trainDataVecs, train["sentiment"] )

    # Test & extract results
    result = forest.predict(testDataVecs)[:,1]
    
    # Write the test results
    output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
    output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )
    print "Wrote Word2Vec_AverageVectors.csv"
