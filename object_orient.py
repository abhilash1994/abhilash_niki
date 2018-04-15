#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import nltk
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection.univariate_selection import SelectKBest, chi2
from sklearn.linear_model import SGDClassifier
import sys
reload(sys)
sys.setdefaultencoding('UTF8')


class MyClassifier(object):
    def __init__(self):
        self.stopwords = set(w.rstrip() for w in open('stopwords_pruned.txt'))
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = self.vectorizer()
        self.label_encoder = self.label_encoder()
        self.svm_classifier = self.svm_classifier()
        self.feature_selector = self.feature_selector()

    def my_tokenizer(self, s):
        s = s.lower()
        tokens = nltk.tokenize.word_tokenize(s)
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        tokens = [t for t in tokens if t not in self.stopwords]
        return tokens

    def classify(self, questions):
        questions = [questions]
        temp_list = []
        for q in questions:
            temp_list.append(' '.join(self.my_tokenizer(q)))
        test_features = self.vectorizer.transform(temp_list).toarray()
        test_features = self.feature_selector.transform(test_features)
        prediction = self.svm_classifier.predict(test_features)
        return (list(self.label_encoder.inverse_transform(prediction))[0].strip())

    def label_encoder(self):
        return LabelEncoder()

    def vectorizer(self, norm='l2'):
        return TfidfVectorizer(
            norm=norm, min_df=0.001, max_features=5000,
            ngram_range=((1, 3)), sublinear_tf=True
        )

    def feature_selector(self):
        return SelectKBest(chi2, k=140)

    def svm_classifier(self):
        return SGDClassifier(
            loss='hinge', penalty='l2', n_iter=5,
            random_state=42
        )

    def pipeline(self):
        df = pd.read_csv('LabelledData.txt', sep=r',,,', header=None)
        self.label_encoder.fit(df[1])
        y = self.label_encoder.transform(df[1])

        questions = []
        for question in df[0]:
            temp = self.my_tokenizer(question)
            temp = ' '.join(temp)
            questions.append(temp)

        self.vectorizer.fit(questions)
        x = self.vectorizer.transform(questions).toarray()
        x = self.feature_selector.fit_transform(x, y)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        self.svm_classifier.fit(x_train, y_train)


if __name__ == '__main__':
    classifier = MyClassifier()
    classifier.pipeline()
    while(1):
        test_qn = raw_input("Enter your Question\n")
        print ("Class : %s" % (classifier.classify(test_qn)))
