#This will be our first shot at using natural language to analyze sentiment
import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize, sent_tokenize
import re 
from nltk.corpus import stopwords, brown 
from nltk.text import Text 
import string 
from nltk.stem import PorterStemmer
from nltk import RegexpParser
import xlrd 
from nltk_sentiment_1 import * 
from nltk.corpus import conll2000
from nltk_helpers import *
#from nltk_sentiment import *
import pickle 
from pickle import dump
from nltk.chunk import tree2conlltags
from helpers_2 import * 
#nltk.download('averaged_perceptron_tagger')
ps = PorterStemmer()
stop_words = list(set(stopwords.words('english')))

path = "C:/Users/aarondardik/Desktop/NLP/test_and_train_scoring/document_scores/score_list.xlsx"

simple_grammar = r"""
    NP: {<DT>*<JJ>*<CD>?<NN.*>*<IN>+<NN.*>+}
        {<DT | PP | PRP | PRP\$>?<JJ.*>*<CD>?<NN.*>+<POS>?<CD>?<NN.*>+}
        {<\$><CD>}
        

"""
entry_words, entry_sentences = excel_to_words_and_sentences(path)

#Note this function returns two lists - words tagged by entry and by sentence
def words_tagged_two_ways():
    a, b = excel_to_words_and_sentences(path) #Here we return two lists - the first a list of lists of words
    #in each entry, the second a list of lists of sentences in each entry - 
    #each of the two tagging methods works on each of these respectively 

    #The following uses a function from helpers to return a list, whose Kth sublist is the 
    #words in the Kth entry, along with their POS tags.
    words_tagged_by_entry = tag_list_of_words(a)
    #This function, is similar to the above but returns a list whose Kth sublist is the words 
    #in the Kth SENTENCE, along with their POS tags.
    words_tagged_by_sentence = tag_list_of_sentences(b)
    return words_tagged_by_entry, words_tagged_by_sentence

#This function returns two lists, each consisting of a TUPLE, where tuple[0] is a list of
#words or sentences (first or second output respectively) and tuple[1] is in {-1, 0, 1}
def scores_tagged_to_words_and_sentences():
    a, b = excel_to_words_and_sentences(path)
    scores = bucket_scores(read_in_excel_scores(path))
    #Here we create a list whose Kth entry is a tuple, the first item in the tuple is a list of
    #words in the Kth entry, the second item is the score associated to said entry
    combined_words = tag_scores_to_words(a, scores) 
    combined_sentences = tag_scores_to_words(b, scores)
    return combined_words, combined_sentences

def find_tickers_in_entries():
    #Returning to work on this if need be...
    a, b = excel_to_words_and_sentences(path)
    return None

#This function returns two lists, the first is a list of all alphanumeric tokens
#in our training set (from excel). The second is a list of all alphanumeric tokens
#in our training set modulo stop_words 
def learning_corpus_words():
    a, b = excel_to_words_and_sentences(path)
    list_ = []
    list_without_stopwords = []
    for entry in a:
        for word in entry:
            if word.isalnum():
                list_.append(word.lower())
                if not(word in stop_words):
                    list_without_stopwords.append(word)
    return list_, list_without_stopwords



corpus1, corpus2 = learning_corpus_words()
score_words, score_sentences = scores_tagged_to_words_and_sentences()

#all_words here is a dictionary, with keys in corpus2 and values being the number of occurances
all_words = nltk.FreqDist(corpus2)
def sort_freq_dist(dictionary):
    return {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1])}
def top_words(dictionary, cutoff):
    sorted_dict = sort_freq_dist(dictionary)
    keylist = list(sorted_dict.keys())
    new_dict = {}
    for k in keylist[-cutoff:]:
        new_dict[k] = sorted_dict[k]
    return new_dict


top = top_words(all_words, 3000)
top_words_list = list(top.keys())
def find_features(document):
    words = set(document)
    features = {}
    for w in top_words_list:
        features[w] = (w in words)

    return features


featureset = [(find_features(item[0]), item[1]) for item in score_words]
numfeatures = len(featureset)
training_set = featureset[:int(0.9 * numfeatures)]
testing_set = featureset[int(0.9*numfeatures):]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print(nltk.classify.accuracy(classifier, testing_set)*100)

filename_NB_classifier_0 = 'NB_classifier_0'
outfile = open(filename_NB_classifier_0, 'wb')
pickle.dump(classifier, outfile)
outfile.close()




#filename = 'chunker_12_pickle'
#infile = open(filename, 'rb')
#chunker_v3 = pickle.load(infile)
#infile.close()
#entry_tag, sentence_tag = words_tagged_two_ways()
#tree_list = []
#for item in entry_tag:
#    tree_list.append(chunker_v3.parse(item))
#it1 = tree2conlltags(tree_list[2])




#^^^^^At this point we are able to read in the excel data, tag the words with POS tags
#and chunk the data with our chunker. 



train_sentences = conll2000.chunked_sents('train.txt', chunk_types = ['NP'])
#chunker1 = ConsecutiveNPChunker(train_sentences)
#featuresets = [[document_features_binary(doc, corpus), score] for doc, score in zip(a, scores)]
#train_set, test_set = featuresets[:int(len(scores) / 2)], featuresets[int(len(scores) / 2):]
#classifier = nltk.NaiveBayesClassifier.train(train_set)
#print(nltk.classify.accuracy(classifier, test_set))








