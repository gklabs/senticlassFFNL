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
import sys, os
import operator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from itertools import islice
from nltk.corpus import stopwords
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

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
    
    def removedstopwords(tweettokens):
        return [word for word in tweettokens if word not in stopwords.words('english')]
    
    #remove the urls. This step is done as they are scored high by TFIDF due to their infrequency
    def removeUrls(tweettokens):
        newtweettokentext = []
        for token in tweettokens:
            if (not re.compile(r'^//t.co/').search(token)) and (token != 'http' and token != 'https'):
                newtweettokentext.append(token)
        return newtweettokentext
    
    cleantweet= []
    for doc in df.text:
        cleantweet.append(cleantweettext(doc))
    df.text= cleantweet
    
    tokentweet=[]    
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
    
    #removing stop words
    nostopwordtweet=[]
    for doc in df.text:
        nostopwordtweet.append(removeUrls(removedstopwords(doc)))
    df.text =nostopwordtweet
       
    #stemming
    stemtweets=[]
    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("english", ignore_stopwords=False)
    
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
        
    return df_stemmed,df_unstemmed


# initialize count vectorizer
def dummy_fun(doc):
    return doc

def getrep(df, rep):
    if rep == 'tfidf': #TF IDF Representation
        #create TF IDF vector using NLTK
        freqdict={}
        counter = 1
        #compute term frequency
        for tweet in df.text:
            tf={}
            counter += 1
            for word in tweet:
                if word in tf:
                    tf[word]+=1
                else:
                    tf[word]= 1
            freqdict[counter]= tf

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

        for word in docufreq:
            invdocufreq[word]= math.log(N/docufreq[word],10)

        '''
        sorted_x = dict(sorted(docufreq.items(), key=lambda kv: kv[1], reverse = True))
        print("20 Highest occuring elements from document frequency matrix ")
        print(list(islice(sorted_x.items(), 20)))
        sorted_x = dict(sorted(invdocufreq.items(), key=lambda kv: kv[1], reverse = True))
        print("20 Highest value elements from inverse document frequency matrix ")
        print(list(islice(sorted_x.items(), 500)))
        '''
        
        TFIDF_dataframe = pd.DataFrame(0,index= list(set(freqdict.keys())),columns= Vocab)
        print(TFIDF_dataframe.shape)
        
        for wid,info in freqdict.items():
            for keyword in info:
                if info[keyword] > 0:
                    tfscore = 1+math.log(info[keyword],10)
                else:
                    tfscore = 0
                TFIDF_dataframe.at[wid,keyword]= tfscore * invdocufreq[keyword]

        TFIDFmatrix= TFIDF_dataframe.values
        TFIDF_dataframe.to_csv("trainTFIDFdf.csv")
        return TFIDF_dataframe,Vocab,TFIDFmatrix, invdocufreq
    
def transformtest(refDF, testdf, invdocufreq, trainVocab):    
    freqdict={}
    #compute term frequency
    for tweet in testdf.text:
        tf={}
        for word in tweet:
            if word in tf:
                tf[word]+=1
            else:
                tf[word]= 1
        freqdict[" ".join(tweet)[:15]]= tf
        
    testTFIDFdf = pd.DataFrame(0, index = list(set(freqdict.keys())), columns= list(refDF.columns))
    
    for wid,info in freqdict.items():
        for keyword in info:
            if (keyword in trainVocab):
                if info[keyword] > 0:
                    tfscore = 1+math.log(info[keyword],10)
                else:
                    tfscore = 0
                testTFIDFdf.at[wid,keyword]= tfscore * invdocufreq[keyword]    
    
    return testTFIDFdf
    

def main():
    os.chdir("D:\\Spring 2020\\assignments\\senticlassFFNL-master\\senticlassFFNL")
    '''# print command line arguments
    train= get_data("tweet\\train") #get_data(sys.argv[1])
    test= get_data("tweet\\test")  #get_data(sys.argv[2])
    
    print("cleaning data")
    clean_train_stem,clean_train_nostem= clean(train)
    #clean_test_stem, clean_test_nostem= clean(test)
    print("cleaning done")
    
    print('shape of stem', clean_train_stem.shape)
    print('shape of non stem', clean_train_nostem.shape)
        
    print("create vectors")
    traindf_stem_tfidf, stem_vocab_tf, trainvector_stem_tfidf, ts_vectorizer= getrep(clean_train_stem, 'tfidf')
    #traindf_nostem_tfidf, nostem_vocab_tf, trainvector_nostem_tfidf, tn_vectorizer= getrep(clean_train_nostem, 'tfidf')

    #test_stem_tfidf_df = transformtest(traindf_stem_tfidf, clean_test_stem, ts_vectorizer,stem_vocab_tf)
    #test_nostem_tfidf_df = transformtest(traindf_nostem_tfidf, clean_test_nostem, tn_vectorizer, nostem_vocab_tf)
    #test_stem_tfidf_df.to_csv("testTFIDFdf_s.csv")
    #test_nostem_tfidf_df.to_csv("testTFIDFdf_n.csv")
    '''
    train= get_data("tweet\\train")
    test= get_data("tweet\\test")
    traindf_stem_tfidf = pd.read_csv("D:\\Spring 2020\\assignments\\senticlassFFNL-master\\senticlassFFNL\\trainTFIDFdf.csv")
    testdf_stem_tfidf = pd.read_csv("D:\\Spring 2020\\assignments\\senticlassFFNL-master\\senticlassFFNL\\testTFIDFdf_s.csv")
    '''
    def dummy_fun(doc):
        return doc
    
    clean_train_stem,clean_train_nostem= clean(train)
    tfidf_vectorizer = TfidfVectorizer(analyzer='word', tokenizer=dummy_fun, preprocessor=dummy_fun, token_pattern=None)    
    traindf_stem_tfidf = tfidf_vectorizer.fit_transform(clean_train_stem.text)
    
    traindf_stem_tfidf.toarray()
    
    X_train, X_val, y_train, y_val = train_test_split(traindf_stem_tfidf, train.Sentiment, test_size=0.33, random_state=42)
    train_data = trainData(torch.FloatTensor(X_train.toarray()), torch.FloatTensor(y_train.values))
    val_data = testData(torch.FloatTensor(X_val.toarray()))#, torch.FloatTensor(y_val.values))
    '''
    X_train, X_val, y_train, y_val = train_test_split(traindf_stem_tfidf.values[:,1:], train.Sentiment, test_size=0.33, random_state=42)
    
    y_train = torch.from_numpy(y_train.values.reshape((y_train.shape[0],-1))).float()
    y_val = torch.from_numpy(y_val.values.reshape((y_val.shape[0],-1))).float()
    
    
    model = FeedForwardNeuralNetwork(X_train.shape[1])  # Feed forward neural network with x.shape[1] dimensions
    learning_rates = [0.1, 0.05, 0.0001]
    for l in learning_rates:
        model.train()    
        optimizer = optim.SGD(model.parameters(), lr=l)  # Optimizing with Stochastic Gradient Descent
        loss = nn.MSELoss()  # Mean Squared Error Loss
        losslist = []   
        x = Variable(torch.from_numpy(X_train))
        #Training
        for epoch in range(50):  # Training Loop
            yhat = model(x.float())  # Compute forward pass
            output = loss(yhat.squeeze(), y_train.squeeze())  # Compute loss
            optimizer.zero_grad()  # Zero all the Gradients
            output.backward()  # Back propagate loss
            optimizer.step()  # Update weights
            losslist.append(output.item())
            print('Iteration: {}. Loss: {}. '.format(epoch, output.item()))#, accuracy, correct, total))
        print('-------minimum loss for learning rate {} is {}-------'.format(l, min(losslist)))
    
    #Validation
    model.eval()
    x = Variable(torch.from_numpy(X_val))
    y_pred = model(x.float())
    val_loss = loss(y_pred.squeeze(), y_val.squeeze()) 
    correct = (torch.round(y_pred) == y_val).sum()
    print("validation accuracy ", 100 * correct /y_pred.shape[0])
    print('validation loss ' , val_loss.item())

class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, dims):
        super(FeedForwardNeuralNetwork, self).__init__()
        self.L1 = nn.Linear(dims,20)
        self.L2 = nn.Linear(20, 20)
        self.L3 = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        a1 = self.sigmoid(self.L1(x))
        a2 = self.sigmoid(self.L2(a1))
        a3 = self.sigmoid(self.L3(a2))
        return a3


if __name__ == "__main__":
    main()










