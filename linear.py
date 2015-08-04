#ref:https://www.kaggle.com/c/word2vec-nlp-tutorial/forums/t/11261/beat-the-benchmark-with-shallow-learning-0-95-lb
import os
from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

import pandas as pd
import numpy as np
import pickle

train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'labeledTrainData.tsv'), header=0, \
                delimiter="\t", quoting=3)
test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'testData.tsv'), header=0, delimiter="\t", \
               quoting=3 )
y = train["sentiment"]  
for i in range(10):
    print y[i]
print "Cleaning and parsing movie reviews...\n"      
traindata = []
for i in xrange( 0, len(train["review"])):
    traindata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(train["review"][i], False)))
testdata = []
for i in xrange(0,len(test["review"])):
    testdata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(test["review"][i], False)))
print 'vectorizing... ', 
for i in range(10):
    train["review"][i] 
tfv = TfidfVectorizer(min_df=4,  max_features=None, 
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words = 'english')
X_all = traindata + testdata
lentrain = len(traindata)

print "fitting pipeline... ",
tfv.fit(X_all)
X_all = tfv.transform(X_all)

pickle.dump(X_all, open("x_all", "w"))

X = X_all[:lentrain]
X_test = X_all[lentrain:]
model = LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                         C=1, fit_intercept=True, intercept_scaling=1.0, 
                         class_weight=None, random_state=None)

print "Retrain on all training data, predicting test labels...\n"
model.fit(X,y)
result = model.predict_proba(X_test)[:,1]
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )

# Use pandas to write the comma-separated output file
output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_model_SVM.csv'), index=False, quoting=3)
print "Wrote results to Bag_of_Words_model.csv"

