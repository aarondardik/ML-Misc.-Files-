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
from nltk.corpus import stopwords
from nltk.text import Text 
import string 
from nltk.stem import PorterStemmer
from nltk import RegexpParser
import xlrd 
from nltk_sentiment_1 import *
from nltk.corpus import conll2000


path = "C:/Users/aarondardik/Desktop/NLP/test_and_train_scoring/document_scores/score_list.xlsx"
#this takes in a list (NOT A LIST OF LISTS) and returns the list with each element 
#tagged per its part of speech - note this makes sense only for a list of words 
def part_of_speech_tagging(word_list):
    return nltk.pos_tag(word_list)
#This function can be used on either an "un-cleaned" or a "cleaned" sentence list
#Note that this function returns a list of lists - to get a single, unnested word list
#we should use the "nested list to list" function
def convert_sentence_list_to_word_list(sentence_list):
    return [word_tokenize(sentence) for sentence in sentence_list]

#This function will generally be used on a list generated by the preceding function,
#"convert sentence list to word list" to give us an unnested list of word for further 
#processing at the word level
def nested_list_to_list(list1):
    unnested = []
    for sublist in list1:
        for item in sublist:
            unnested.append(item)
    return unnested
#///////////////////////////////////////////////////////////
#Note this function reads in a list of bodies from our train / test set
def read_in_excel_entries(file_path):
   workbook = xlrd.open_workbook(file_path)
   sheet = workbook.sheet_by_index(0)
   body_list = []
   for i in range(sheet.nrows):
       body_list.append(sheet.cell_value(i, 1) + '.')
   return body_list

#Note this function reads in scores from our train / test set
def read_in_excel_scores(file_path):
    workbook = xlrd.open_workbook(file_path)
    sheet = workbook.sheet_by_index(0)
    fun_scores = []
    for i in range(sheet.nrows):
        fun_scores.append(sheet.cell_value(i, 2))
    return fun_scores

#Note this function reads in a list of scores and returns a list of values
#+1, if the score is positive, -1 if negative
def bucket_scores(score_list):
    binary_score_list = []
    for item in score_list:
        if (item < 1):
            binary_score_list.append(-1)
        else:
            binary_score_list.append(1)
    return binary_score_list

#This method takes a list and returns a string with whitespace between each list element
def list_to_string(l):
    return ' '.join(l)


def convert_doc_to_sentences(doc):
    return sent_tokenize(doc)
#this function takes in our excel body entries and returns two lists
#BOTH lists are lists of lists. In the first list, each sub-list corresponds
#to an entry in the excel sheet, with items as WORDS in the entry. In the 
#second list, each sub-list corresponds to an entry in the excel sheet
#with items as SENTENCES in the entry
def excel_to_words_and_sentences(file_path):
    #we will read in entries, and for each entry parse that entry into sentences...
    x = read_in_excel_entries(file_path)
    #Note list of entries is a list of lists, each sub-list corresponds to one entry 
    #and the items of each sub-list are the sentences in that entry
    list_of_entries = []
    for entry in x:
        y = convert_doc_to_sentences(entry)
        list_of_entries.append(y)

    words_per_entry = []
    for item in list_of_entries:
        words = convert_sentence_list_to_word_list(item)
        words = nested_list_to_list(words)
        words_per_entry.append(words)
    return words_per_entry, list_of_entries



#Note this function takes in a list of lists, where the Jth sublist
#read in corresponds to the words in the Jth entry. It outputs a list
#of lists, whose Kth sublist is a list of words in the Kth entry, 
#tagged with their POS tags.
def tag_list_of_words(top_level_list):
    tagged_list = []
    for item in top_level_list:
        tagged_list.append(part_of_speech_tagging(item))
    return tagged_list

#Note this function reads in a list of lists, where the Jth sublist
#read in consists of the sentences of the Jth entry. It returns a list
#of lists, whose Kth sublist consists of the words of the Kth sentence
#tagged with their POS tags
def tag_list_of_sentences(top_level_list):
    tagged_sentences = []
    for entry in top_level_list:
        words_in_entry = convert_sentence_list_to_word_list(entry)
        for sentence in words_in_entry:
            pos = part_of_speech_tagging(sentence)
            tagged_sentences.append(pos)
    return tagged_sentences


#This function will take in a list of lists of words in an entry as well
#as a list of scores per entry, and combine it into a list of tuples
#where the Kth tuple consists of 1) the words in the Kth entry
#and 2) the Kth entry's fundamental score
def tag_scores_to_words(word_list, scores_list):
    combined_list = []
    for i in range(len(word_list)):
        combined_list.append((word_list[i], scores_list[i]))
    return combined_list 

#Note this function takes in a list of word lists, and returns a list of words
def all_words(list_of_lists):
    word_list = []
    for sublist in list_of_lists:
        for item in sublist:
            word_list.append(item.lower())
    return word_list

#This reads in a document, and a list of words in the corpus and
#returns a dict with keys being the words in the corpus
#and values True or False if the word is in doc or not
def document_features_binary(document, corpus):
    corpus_words = set(corpus)
    features = {}
    for word in corpus_words:
        features['contains({})'.format(word)] = (word in document)
    return features 
