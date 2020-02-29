'''
Author: Giridhar R, Haritha G
Description: AIT 726 Homework 2
Command to run the file:
(python senticlassFFNN.py [train folder location] [test folder location] )

Detailed Procedure:

Start:
getdata()
cleandata()
vectorize()


'''
import pandas as pd
from collections import defaultdict
from pathlib import Path
import nltk as nl
from nltk.tokenize import word_tokenize
import re
from nltk.tokenize.casual import TweetTokenizer
import numpy as np
import math
import sys
import operator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from itertools import islice

#nl.download('punkt')

# pass path as train and test folder

#stdoutorigin = sys.stdout
#sys.stdout = open("D:\\Spring 2020\\assignments\\sentiment_classification\\log.txt", "w")
################ Functions######################
def get_data(path):
    pospath = path + '/positive'
    # create list to store text
    results = defaultdict(list)

    # loop through files and append text to list
    for file in Path(pospath).iterdir():
        with open(file, "r", encoding="utf8") as file_open:
            results["text"].append(file_open.read())

    # read the list in as a dataframe
    df_pos = pd.DataFrame(results)

    #set directory path
    my_dir_path_neg = path + '/negative'

    # create list to store text
    results_neg = defaultdict(list)

    # loop through files and append text to list
    for file in Path(my_dir_path_neg).iterdir():
        with open(file, "r", encoding="utf8") as file_open:
            results_neg["text"].append(file_open.read())
    # read the list in as a dataframe
    df_neg = pd.DataFrame(results_neg)
    df_neg.head()

    #add sentiment to both datasets and then combine them for test data 1 for positive and 0 for negative
    df_pos['Sentiment']=1
    df_neg['Sentiment']=0
    frames = [df_pos, df_neg]
    df = pd.concat(frames)

    # increase column width to see more of the tweets
    pd.set_option('max_colwidth', 140)

    # reshuffle the tweets to see both pos and neg in random order
    df = df.sample(frac=1).reset_index(drop=True)

    # explore top 5 rows
    #df.head(5)
    return df

def clean(df):

    # Remove any markup tags (HTML), all the mentions of handles(starts with '@') and '#' character
    def cleantweettext(raw_html):
        pattern = re.compile('<.*?>')
        cleantext = re.sub(pattern, '', raw_html)
        cleantext = " ".join(filter(lambda x:x[0]!='@', cleantext.split()))
        cleantext = cleantext.replace('#', '')
        return cleantext

    def removeat(text):
        atlist=[]
        for word in text:
            pattern = re.compile('^@')
            if re.match(pattern,word):
                #cleantext1 = re.sub(pattern, word[1:], word)
                atlist.append(word[1:])
            else:
                atlist.append(word)
        return atlist

    def tolower(text):
        lowerlist=[]
        for word in text:
            pattern = re.compile('[A-Z][a-z]+')
            if re.match(pattern,word):
                cleantext1 = re.sub(pattern, word.lower(), word)
                lowerlist.append(cleantext1)
            else:
                lowerlist.append(word)
        return lowerlist

    #removing stop words
    #def stopwords(text):

    cleantweet= []
    for doc in df.text:
        cleantweet.append(cleantweettext(doc))


    tokentweet=[]
    df.text= cleantweet
    for doc in df.text:
        tokentweet.append(TweetTokenizer().tokenize(doc))
    df.text= tokentweet

    removeattweet=[]
    for doc in df.text:
        removeattweet.append(removeat(doc))
    df.text =removeattweet

    lowertweet=[]
    for doc in df.text:
        lowertweet.append(tolower(doc))
    df.text = lowertweet

    tweets=[]
    for x in df.text:
        tweet = ''
        for word in x:
            tweet += word+' '
        tweets.append(word_tokenize(tweet))
    df.text= tweets

    #stemming
    stemtweets=[]
    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("english", ignore_stopwords=False)
    #ps= PorterStemmer()
    for x in df.text:
        stemtweet=''
        for word in x:
            stemtweet=stemtweet+stemmer.stem(word)+' '
        stemtweets.append(word_tokenize(stemtweet))
    df['stemmed']=stemtweets

    df_unstemmed = pd.DataFrame()
    df_unstemmed['text'] = df['text']
    df_unstemmed['Sentiment'] = df['Sentiment']
    df_stemmed = pd.DataFrame()
    df_stemmed['text'] = df['stemmed']
    df_stemmed['Sentiment'] = df['Sentiment']
    
    ### Finalize both the stemmed and unstemmed dataframes
    #df_unstemmed = df.drop(['stemmed'], axis=1)
    #df_unstemmed.head()

    # create a df with stemmed text
    #df_stemmed = df.drop(['text'], axis=1)
    
    return df_stemmed,df_unstemmed


# initialize count vectorizer
def dummy_fun(doc):
    return doc

def getrep(df, rep):
    if rep == 'binary':
        vectorizer = CountVectorizer(binary = True, analyzer='word', tokenizer=dummy_fun, preprocessor=dummy_fun, token_pattern=None)
        text = df.text
        vectorizer.fit(text)
        freqVocab = vectorizer.vocabulary_
        train_vector = vectorizer.transform(text)
        #Create bigdoc that contains words in V, their corresponding frequencies for each class

        #1.Transform pos and neg tweets into seprate vectors
        train_pos_vector1 = vectorizer.transform(df[df['Sentiment']==1]['text'])
        train_neg_vector1 = vectorizer.transform(df[df['Sentiment']==0]['text'])

        #2. column sum of vectors(word per column)
        sum_pos = train_pos_vector1.sum(axis = 0)
        sum_neg = train_neg_vector1.sum(axis = 0)

        #3. Initialize bigdoc as a dataframe
        bigdoc = pd.DataFrame(index = list(set(freqVocab.keys())), columns = ['pos', 'neg'])

        #4. get the corresponding frequency from the above matrx and set it to bigdoc
        for word in freqVocab.keys():
            index = freqVocab.get(word)
            bigdoc.at[word, 'pos'] = sum_pos[:, index].item()
            bigdoc.at[word, 'neg'] = sum_neg[:, index].item()
        return bigdoc, freqVocab, train_vector, vectorizer

    elif rep == 'freq': #for frequency representation

        vectorizer = CountVectorizer(analyzer='word', tokenizer=dummy_fun, preprocessor=dummy_fun, token_pattern=None)          
        text = df.text
        vectorizer.fit(text)
        freqVocab = vectorizer.vocabulary_
        train_vector = vectorizer.transform(text)
        #Create bigdoc that contains words in V, their corresponding frequencies for each class

        #1.Transform pos and neg tweets into seprate vectors
        train_pos_vector1 = vectorizer.transform(df[df['Sentiment']==1]['text'])
        train_neg_vector1 = vectorizer.transform(df[df['Sentiment']==0]['text'])

        #2. column sum of vectors(word per column)
        sum_pos = train_pos_vector1.sum(axis = 0)
        sum_neg = train_neg_vector1.sum(axis = 0)

        #3. Initialize bigdoc as a dataframe
        bigdoc = pd.DataFrame(index = list(set(freqVocab.keys())), columns = ['pos', 'neg'])

        #4. get the corresponding frequency from the above matrx and set it to bigdoc
        for word in freqVocab.keys():
            index = freqVocab.get(word)
            bigdoc.at[word, 'pos'] = sum_pos[:, index].item()
            bigdoc.at[word, 'neg'] = sum_neg[:, index].item()

        return bigdoc, freqVocab, train_vector, vectorizer

    elif rep == 'tfidf': #TF IDF Representation
        #create TF IDF vector using NLTK
        freqdict={}
        #compute term frequency
        for tweet in df.text:
            tf={}
            for word in tweet:
                if word in tf:
                    tf[word]+=1
                else:
                    tf[word]= 1
            freqdict[" ".join(tweet)[:15]]= tf

        #compute inverse document frequency
        N= len(df) # total number of documents
        # compute number of documents with a word w
        invdocufreq={}

        def createvocab(df):
            V=[]
            for tweet in df.text:
                for keyword in tweet:
                    if keyword  in V:
                        continue
                    else :
                        V.append(keyword)
            return V

        def wordintweet(text,keyword):
            for word in text:
                if word == keyword:
                    return 1
            return 0
        
        Vocab = createvocab(df)
        Vocab= sorted(Vocab)
        docufreq= {el:0 for el in Vocab}
        invdocufreq= {el:0 for el in Vocab}

        for word in Vocab:
            for tweet in df.text:
                if wordintweet(tweet, word) == 1:
                    docufreq[word]+=1

        for word in  docufreq:
            invdocufreq[word]= math.log(N/docufreq[word],10)

        
        sorted_x = dict(sorted(docufreq.items(), key=lambda kv: kv[1], reverse = True))
        print("20 Highest occuring elements from document frequency matrix ")
        print(list(islice(sorted_x.items(), 20)))
        sorted_x = dict(sorted(invdocufreq.items(), key=lambda kv: kv[1], reverse = True))
        print("20 Highest value elements from inverse document frequency matrix ")
        print(list(islice(sorted_x.items(), 20)))
        TFIDF_dataframe = pd.DataFrame(0,index= list(set(freqdict.keys())),columns= Vocab)

        print(TFIDF_dataframe.shape)
        
        #print(freqdict)

        for wid,info in freqdict.items():
            for keyword in info:
                if word== keyword:
                    TFIDF_dataframe.at[wid,keyword]= info[keyword]* invdocufreq[keyword]


        #TFIDFmatrix =  np.zeros(len(df),len(Vocab)) #initialize
        TFIDFmatrix= TFIDF_dataframe.values
        print(TFIDFmatrix)
        return TFIDF_dataframe,Vocab,TFIDFmatrix, None
    


def main():

    # print command line arguments
    train= get_data(sys.argv[1])
    test= get_data(sys.argv[2])
    
    print("cleaning data")
    clean_train_stem,clean_train_nostem= clean(train)
    clean_test_stem, clean_test_nostem= clean(test)
    print("cleaning done")
    #print(clean_train_stem.head(5))
    #print(clean_train_nostem.head(5))
    
    print("create vectors")
    traindf_stem_tfidf, stem_vocab_tf, trainvector_stem_tfidf, ts_vectorizer= getrep(clean_train_stem, 'tfidf')
    traindf_nostem_tfidf, stem_vocab_tf, trainvector_nostem_tfidf, tn_vectorizer= getrep(clean_train_nostem, 'tfidf')

            

if __name__ == "__main__":
    main()













