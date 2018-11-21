#Author :: Sairam Bade
##for getting unique words
#cut -d"," -f2 file | grep -o -E '\w+' | sort -u -f

import re
import os
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import numpy as np 
import pandas as pd 
import pickle
from nltk.stem import PorterStemmer
from collections import Counter
from sklearn.metrics import f1_score
from nltk.stem import WordNetLemmatizer

from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import  FeatureUnion, Pipeline ,make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer
from sklearn.metrics import f1_score,accuracy_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
import csv

def decontracted(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    phrase = re.sub(r"gimme","give me", phrase)
    phrase = re.sub(r"cancelled","canceled",phrase)
    return phrase

#reading in training data and removing special characters
def process_data(filepath):
	df = pd.read_csv(filepath)
	text_tr = []
	label_tr = []
	for index, row in df.iterrows():
		decontract_text = decontracted(row['text'])
		clean_text = re.sub('([.,!?-_\'])','', decontract_text)
		text_tr.append(clean_text)
		label_tr.append(row['label'])
	##remove stopwords and perform stemming
	##stemming creates words not present in word2vec representation
	stop_words = set(stopwords.words('english'))
	ps = PorterStemmer()
	lemmatizer = WordNetLemmatizer()
	text_tr_sw = []
	text_tr_act = []
	text_tr_lemma = []
	for index,sent in enumerate(text_tr):
		word_tokens = word_tokenize(sent) 
		filtered_sentence = [w for w in word_tokens if not w in stop_words] 
		filtered_sentence1 = [ps.stem(w) for w in filtered_sentence]
		filtered_sentence2 = [lemmatizer.lemmatize(w) for w in filtered_sentence] # gives actual words
		text_tr_sw.append(filtered_sentence1)
		text_tr_lemma.append(filtered_sentence2)
		text_tr_act.append(filtered_sentence)
	return text_tr_act,text_tr_sw,text_tr_lemma,label_tr

text_tr_full,text_tr,text_tr_lemma,label_tr = process_data('ml-task-datasets/data_train.csv')
text_dev_full,text_dev,text_dev_lemma,label_dev = process_data('ml-task-datasets/data_dev.csv')

print('Train ::', Counter(label_tr).most_common(),'\n')
print('Dev ::', Counter(label_dev).most_common(),'\n')

##Analyse the words in train and dev set

##get most frequent words per class
def word_analyse(text_tr_full,label_tr,target):
	customer_text =[]
	for (text,label) in zip(text_tr_full,label_tr):
		if label==target:
			customer_text =  customer_text + text
	return customer_text

# print (Counter(word_analyse(text_tr_full,label_tr,'order')).most_common(10))
# print (Counter(word_analyse(text_tr_full,label_tr,'customer')).most_common(10))
# print (Counter(word_analyse(text_tr_full,label_tr,'shopper')).most_common(10))
# print (Counter(word_analyse(text_tr_full,label_tr,'applicant')).most_common(10))
# print (Counter(word_analyse(text_tr_full,label_tr,'misc')).most_common(10))

# ##on full data train
# #number of words after stop wrod removal = 3405
# train_words = Counter(word_analyse(text_tr_full,label_tr,'order')+word_analyse(text_tr_full,label_tr,'customer')
# 	+ word_analyse(text_tr_full,label_tr,'shopper') + word_analyse(text_tr_full,label_tr,'applicant') 
# 	+ word_analyse(text_tr_full,label_tr,'misc')).keys()


# print (Counter(word_analyse(text_dev_full,label_dev,'order')).most_common(10))
# print (Counter(word_analyse(text_dev_full,label_dev,'customer')).most_common(10))
# print (Counter(word_analyse(text_dev_full,label_dev,'shopper')).most_common(10))
# print (Counter(word_analyse(text_dev_full,label_dev,'applicant')).most_common(10))
# print (Counter(word_analyse(text_dev_full,label_dev,'misc')).most_common(10))
# ##on full data test
# #number of words after stop wrod removal = 2446
# dev_words = Counter(word_analyse(text_dev_full,label_dev,'order')+word_analyse(text_dev_full,label_dev,'customer')
# 	+ word_analyse(text_dev_full,label_dev,'shopper') + word_analyse(text_dev_full,label_dev,'applicant') 
# 	+ word_analyse(text_dev_full,label_dev,'misc')).keys()

# ##OOVS in dev set
# print([item for item in dev_words if item not in train_words])

#combine words to a sentence in text_tr and text_dev
text_tr_s = []
text_dev_s = []
for words_list in text_tr:
	text_tr_s.append(' '.join(words_list))
for words_list in text_dev:
	text_dev_s.append(' '.join(words_list))

text_tr_full_s = []
text_dev_full_s = []
for words_list in text_tr_full:
	text_tr_full_s.append(' '.join(words_list))
for words_list in text_dev_full:
	text_dev_full_s.append(' '.join(words_list))

text_tr_lemma_s = []
text_dev_lemma_s = []
for words_list in text_tr_lemma:
	text_tr_lemma_s.append(' '.join(words_list))
for words_list in text_dev_lemma:
	text_dev_lemma_s.append(' '.join(words_list))
##Naive Bayes Classifier
# text_clf_NB = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),('clf', MultinomialNB(alpha=1)),])

# text_clf_NB = text_clf_NB.fit(text_tr_s,label_tr)

# predicted = text_clf_NB.predict(text_dev_s)
# # print("Accuracy train:: ",np.mean(text_clf_NB.predict(text_tr_s) == label_tr))
# print("Accuracy dev:: ",np.mean(predicted == label_dev))
# print(f1_score(text_dev_s, predicted, average='macro'))

##logistic regression

# text_clf_logistic = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1))),('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),('clf', LogisticRegression(C=1, penalty='l1', random_state=0)),])

# # ## Grid search for parameters
# # parameters = {"clf__C":[0.01,0.1,0.5,1,5,10]}

# # gs_clf_svm = GridSearchCV(estimator= text_clf_logistic, param_grid=parameters)
# # gs_clf_svm = gs_clf_svm.fit(text_dev_s, label_dev)


# # print(gs_clf_svm.best_score_)
# # print(gs_clf_svm.best_params_)

# text_clf_logistic = text_clf_logistic.fit(text_tr_s,label_tr)

# predicted = text_clf_logistic.predict(text_dev_s)

# print("Accuracy dev logistic:: ",np.mean(predicted == label_dev))


##SVM Classifier

# class_wts = {'order': 1,'customer':1 ,'shopper':5,'applicant':10,'misc':10}
# text_clf_svm = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),('clf',svm.SVC(kernel='rbf', C=100, gamma=1,class_weight=class_wts)),])

# text_clf_svm = text_clf_svm.fit(text_tr_s,label_tr)

# predicted_svm = text_clf_svm.predict(text_dev_s)
# print(Counter(predicted_svm).most_common(4))
# print("Accuracy dev SVM:: ",np.mean(predicted_svm == label_dev))


# print("CV starts")

# Grid search for parameters
# parameters = {"clf__C":[1,100,1000],"clf__gamma":[0,0.01,0.001,0.0001,1,100,1000]}

# gs_clf_svm = GridSearchCV(estimator= text_clf_svm, param_grid=parameters)
# gs_clf_svm = gs_clf_svm.fit(text_dev_s, label_dev)


# print(gs_clf_svm.best_score_)
# print(gs_clf_svm.best_params_)

# ##gradient boosting
# text_clf_xgb = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),
# 	('clf',xgb.XGBClassifier(max_depth=10, n_estimators=100, learning_rate=0.1, 
#                                                  gamma=.01, reg_alpha=5, objective='multi:softmax')),])

# text_clf_xgb = text_clf_xgb.fit(text_tr_s,label_tr)
# predicted_xgb = text_clf_xgb.predict(text_dev_s)
# print("Accuracy dev XGB :: ",np.mean(predicted_xgb == label_dev))


## Grid search for parameters
# parameters = {"clf__learning_rate":[0.01,0.1,1,3,10],"clf__reg_alpha":[0.01,0.1,0.5,1,5,10]}

# gs_clf_svm = GridSearchCV(estimator= text_clf_xgb, param_grid=parameters)
# gs_clf_svm = gs_clf_svm.fit(text_dev_s, label_dev)


# print(gs_clf_svm.best_score_)
# print(gs_clf_svm.best_params_)

##use word2vecs

#Dumping word2vecs to a file
from gensim.models import KeyedVectors
filename = 'GoogleNews-vectors-negative300.bin'
if(os.path.isfile('doc2vec_eval_lemma.p')):
	pass
else:
	google_word2vec = KeyedVectors.load_word2vec_format(filename, binary=True)

# 
def get_doc2vec(text_tr_lemma):
	"""gets the averaged word2vecs of all word vectors"""
	output = []
	for words_list in text_tr_lemma:
		word_vecs = []
		for word in words_list:
			try:
				word_vecs.append(google_word2vec[word])
			except Exception:
				pass
		out1 = [sum(i) for i in zip(*word_vecs)]
		output.append(out1)
	return output

# # pickle doc2vecs 

# doc2vec_tr_lemma = get_doc2vec(text_tr_lemma)
# pickle.dump(doc2vec_tr_lemma,open('doc2vec_tr_lemma.p','wb'))
# doc2vec_dev_lemma = get_doc2vec(text_dev_lemma)
# pickle.dump(doc2vec_dev_lemma,open('doc2vec_dev_lemma.p','wb'))
# doc2vec_tr_full = get_doc2vec(text_tr_full)
# pickle.dump(doc2vec_tr_full,open('doc2vec_tr_full.p','wb'))
# doc2vec_dev_full = get_doc2vec(text_dev_full)
# pickle.dump(doc2vec_dev_full,open('doc2vec_dev_full.p','wb'))

with open('doc2vec_tr_lemma.p','rb') as pickle_file:
	doc2vec_tr_lemma = pickle.load(pickle_file)
with open('doc2vec_dev_lemma.p','rb') as pickle_file:
	doc2vec_dev_lemma = pickle.load(pickle_file)
# with open('doc2vec_tr_full.p','rb') as pickle_file:
# 	doc2vec_tr_full = pickle.load(pickle_file)
# with open('doc2vec_dev_full.p','rb') as pickle_file:
# 	doc2vec_dev_full = pickle.load(pickle_file)

# print(len(doc2vec_tr_lemma))
# print(len(doc2vec_dev_lemma))
# print(len(doc2vec_tr_full))
# print(len(doc2vec_dev_full))

class_wts = {'order': 1,'customer':1 ,'shopper':5,'applicant':10,'misc':10}
word2vec_clf_svm = Pipeline([('clf',svm.SVC(kernel='rbf', C=10, gamma=0.0001,class_weight=class_wts)),])
word2vec_clf_svm = word2vec_clf_svm.fit(doc2vec_tr_lemma,label_tr)

predicted_w2v = word2vec_clf_svm.predict(doc2vec_dev_lemma)
#print('misc' in predicted)
print(Counter(predicted_w2v).most_common(5))
print("Accuracy dev word2vec :: ", np.mean(predicted_w2v == label_dev))

# word2vec_clf_logistic = Pipeline([('clf', LogisticRegression(C=0.1, penalty='l1', random_state=0)),])
# word2vec_clf_logistic = word2vec_clf_logistic.fit(doc2vec_tr_full,label_tr)
# predicted = word2vec_clf_logistic.predict(doc2vec_dev_full)
# print("Accuracy dev word2vec :: ", np.mean(predicted == label_dev))

## Grid search for parameters
# parameters = {"clf__C":[1,10,100,1000],"clf__gamma":[0,0.01,0.001,0.0001,1,100,1000],"clf__kernel":['rbf','linear']}

# gs_clf_svm = GridSearchCV(estimator= word2vec_clf_svm, param_grid=parameters)
# gs_clf_svm = gs_clf_svm.fit(doc2vec_dev_full, label_dev)

# print(gs_clf_svm.best_score_)
# print(gs_clf_svm.best_params_)

## word2vec + tf-idf 

# class word2vec_base(BaseEstimator, TransformerMixin):
# 	"""does nothing has to be used in word2vec feature union"""
# 	def fit(self, x, y=None):
# 		return self
# 	def trasform(self,posts):
# 		return get_doc2vec(posts)

# tfidf_vect = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer(use_idf=True,smooth_idf=True)),])

# word2vec_tf_clf_logistic = Pipeline([('feats',FeatureUnion([('word2vec',word2vec_base()),('tfidf',tfidf_vect)])),
#  	('clf', LogisticRegression(C=0.1, penalty='l1', random_state=0)),])
# word2vec_tf_clf_logistic = word2vec_tf_clf_logistic.fit(text_tr_lemma,label_tr)
# predicted = word2vec_clf_tf_logistic.predict(text_dev_lemma)
# print("Accuracy dev word2vec-tfidf :: ", np.mean(predicted == label_dev))

# #Use all best models 
# def vote(tuple):
# 	""" Majority voter or use the SVM output"""
# 	c = Counter(tuple)
# 	value, count = c.most_common()[0]
# 	if count > 1:
# 		return value
# 	else:
# 		return tuple[0]
# prediction_merged = [[s[0],s[1],s[2]] for s in zip(predicted_svm, predicted_xgb, predicted_w2v)]
# prediction_vote = [vote(p) for p in prediction_merged]
# # print(prediction_vote)
# # print(label_dev)
# prediction_vote_np = np.asarray(prediction_vote)
# print("Accuracy dev vote :: ", np.mean(prediction_vote_np == label_dev))


##Bagging
# class_wts = {'order': 1,'customer':1 ,'shopper':5,'applicant':10,'misc':10}
# word2vec_clf_rf = Pipeline([('clf',RandomForestClassifier(n_jobs=4, class_weight=class_wts, criterion='gini',n_estimators=200,max_depth=30,min_samples_split=2,random_state=0)),])
# word2vec_clf_rf = word2vec_clf_rf.fit(doc2vec_tr_lemma,label_tr)

# predicted_rf = word2vec_clf_rf.predict(doc2vec_dev_lemma)
# #print('misc' in predicted)
# print(Counter(predicted_rf).most_common(5))
# print("Accuracy dev word2vec :: ", np.mean(predicted_rf == label_dev))

## Grid search for parameters
# parameters = {"clf__n_estimators":[100,200, 400, 800,1200,1600],"clf__max_depth":[10, 30, 50, 80, 100, None],"clf__min_samples_split":[2, 5, 10]}
# gs_clf_svm = GridSearchCV(estimator= word2vec_clf_rf, param_grid=parameters)
# gs_clf_svm = gs_clf_svm.fit(doc2vec_dev_lemma, label_dev)


# print(gs_clf_svm.best_score_)
# print(gs_clf_svm.best_params_)

###Final processing for eval data.


df = pd.read_csv('ml-task-datasets/data_eval.csv')
text_eval = []
call_id = []
for index, row in df.iterrows():
	decontract_text = decontracted(row['text'])
	clean_text = re.sub('([.,!?-_\'])','', decontract_text)
	text_eval.append(clean_text)
	call_id.append(row['call_id'])
##remove stopwords and perform stemming
##stemming creates words not present in word2vec representation
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
text_eval_lemma = []
for index,sent in enumerate(text_eval):
	word_tokens = word_tokenize(sent) 
	filtered_sentence = [w for w in word_tokens if not w in stop_words] 
	filtered_sentence2 = [lemmatizer.lemmatize(w) for w in filtered_sentence] # gives actual words
	text_eval_lemma.append(filtered_sentence2)

# ##dump docvec to file
if(os.path.isfile('doc2vec_eval_lemma.p')):
	with open('doc2vec_eval_lemma.p','rb') as pickle_file:
		doc2vec_eval_lemma = pickle.load(pickle_file)
else:
	doc2vec_eval_lemma = get_doc2vec(text_eval_lemma)
	pickle.dump(doc2vec_eval_lemma,open('doc2vec_eval_lemma.p','wb'))


predicted_eval = word2vec_clf_svm.predict(doc2vec_eval_lemma)

### write output values 
rows = zip(call_id,predicted_eval)
with open("predictions_eval.csv","w") as f:
	writer = csv.writer(f)
	writer.writerow(["call_id","label"])
	for row in rows:
		writer.writerow(row)


###class wise accuracy calculator
# def class_wise(evals,targets,label):
# 	eval_class = []
# 	target_class = []
# 	for i,j in zip(evals,targets):
# 		if j==label:
# 			target_class.append(j)
# 			eval_class.append(i)
# 	eval_class = np.asarray(eval_class)
# 	return np.mean(eval_class==target_class)

# print(class_wise(predicted_w2v,label_dev,'customer'))
# print(class_wise(predicted_w2v,label_dev,'order'))
# print(class_wise(predicted_w2v,label_dev,'shopper'))
# print(class_wise(predicted_w2v,label_dev,'applicant'))
# print(class_wise(predicted_w2v,label_dev,'misc'))
# print("$$$$$$$$$$$$$$$$")
# print(class_wise(predicted_rf,label_dev,'customer'))
# print(class_wise(predicted_rf,label_dev,'order'))
# print(class_wise(predicted_rf,label_dev,'shopper'))
# print(class_wise(predicted_rf,label_dev,'applicant'))
# print(class_wise(predicted_rf,label_dev,'misc'))