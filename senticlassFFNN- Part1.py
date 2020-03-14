'''
Author: Giridhar R, Haritha G
Description: AIT 726 Homework 2
Command to run the file:
(python senticlassFFNN.py [train folder location] [test folder location] )

Detailed Procedure:

1. import dependencies
2. get data
    Read training data of positive and negative tweets
    Read testing data of positive and negative tweets
3. Clean data
    remove html tags, flight names starting with @, links (strings with //)
    lower case the word if the first letter is uppercase
    remove stop words
    Tokenize the word and emojis using TweetTokenizer on nltk.tokenize.casual 
    Create seperate datasets with stemmed tokens and non stemmed tokens 
4. Vectorization
   Train(Fit - Transform): 
      TF- IDF representation from scratch
      Calculate Term Frequency of every token in every tweet
      Calculate Document Frequency of every token across training data
      Compute Inverse Document Frequency (formula)
      Compute TF*IDF
      Create a Matrix representation of TF*IDF (embedding) and return it.  
5.Build FFNN 
part1: Model building using Feedforward in Pytorch's nn.Module      
      dimensions - Input layer : size of vocab(5639 for stem, 7 for unstemmed)
                   Hidden layer: 20 nodes 
                   Output layer: 1 which is the probability score if the tweet is positive      
      Sigmoid as Activation function for the linear layers
      Initialize weights to random numbers
Part2: Training
      Train the model on the vectorized representation of train data
      Use k-fold cross validation for k=3 to test the generalizability of the model
      Run the training for 2000 epochs and learning rates set = [0.0001, 0.005, 0.01]  
      Initialize the loss function to MSE loss
      Use Stochastic Gradient descent optimizer
      Use batch size = 200
      for each epoch:
          for 1 pass(batch size = 200):
              On train:
              Do Forward propagation to compute the probability 
              Compute training loss
              Backpropagate the error and update weights
              On validation:
              Do Forward propagation to compute the probability 
              Compute validation loss               
          Compute overall training and validation cost for the epoch by taking avaerage of all the losses for each pass
          Compute validation accuracy, Confusion Matrix -------PENDING
      Visualize the loss for each epoch and validation accuracy for each epoch
7. Predict on Test set
   Clean Test Data
   Vector represenation of test:
       Create Term frequncy matrix for all the tokens in test data
       Multipy TF of test data with the IDF representation of train data to get test vector representation
   Use tuned parameters of the final neural network 
   Compute the probabilities for the test
   Print Accuracy
   Print Confusion matrix -------PENDING
            
  ##############################
  
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
from sklearn import preprocessing
import time
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
tqdm.pandas()
from sklearn.metrics import f1_score
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

'''
This method is used to create vector representation for the tokenized data.
Steps:Calculate Term Frequency of every token in every tweet
      Calculate Document Frequency of every token across training data
      Compute Inverse Document Frequency (formula)
      Compute TF*IDF
      Create a Matrix representation of TF*IDF (embedding) and return it.
'''
# initialize count vectorizer
def getrep(df, rep):
    def dummy_fun(doc):
        return doc

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
    counter = 0
    for tweet in testdf.text:
        tf={}
        counter += 1
        for word in tweet:
            if word in tf:
                tf[word]+=1
            else:
                tf[word]= 1
        freqdict[counter]= tf
        
    testTFIDFdf = pd.DataFrame(0, index = list(set(freqdict.keys())), columns= list(refDF.columns))
    #Get the document frequency based on Train IDF
    for wid,info in freqdict.items():
        for keyword in info:
            if (keyword in trainVocab):
                if info[keyword] > 0:
                    tfscore = 1+math.log(info[keyword],10)
                else:
                    tfscore = 0
                testTFIDFdf.at[wid,keyword]= tfscore * invdocufreq[keyword]    
    
    return testTFIDFdf


'''
Training method performs model building, learning parameters from train set and test on validation set
Training runs for several epochs and different learning rates 
'''
def train_pred(model, train_X, train_y, val_X, val_y, epochs=2, batch_size=200):    
    valid_preds = np.zeros((val_X.size(0)))     
    trainloss = []
    testloss = []
    testaccuracy = []
    lrs = [0.0001, 0.005, 0.01]
    for l in lrs:            
        for e in range(epochs):
            start_time = time.time()
                           
            optimizer = optim.SGD(model.parameters(), lr=l)  #Optimizing with Stochastic Gradient Descent
            loss_fn = nn.MSELoss()  # Mean Squared Error Loss
           
            train_tsdata = torch.utils.data.TensorDataset(train_X, train_y)
            valid_tsdata = torch.utils.data.TensorDataset(val_X, val_y)
            
            train_loader = torch.utils.data.DataLoader(train_tsdata, batch_size=batch_size, shuffle=True)
            valid_loader = torch.utils.data.DataLoader(valid_tsdata, batch_size=batch_size, shuffle=False)
          
            model.train()
            avg_loss = 0.
            for x_batch, y_batch in tqdm(train_loader, disable=True):
                y_pred = model(x_batch.float()).requires_grad_(True)
                loss = loss_fn(y_pred.squeeze(), y_batch.float().squeeze())
                #loss.requires_grad = True 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() / len(train_loader)
            trainloss.append(avg_loss)
            
            model.eval()          
            avg_val_loss = 0.
            testacc = 0
            for i, (x_batch, y_batch) in enumerate(valid_loader):
                y_pred = model(x_batch.float()).detach()
                avg_val_loss += loss_fn(y_pred, y_batch.float()).item() / len(valid_loader)
                valid_preds[i * batch_size:(i+1) * batch_size] = y_pred[:, 0].data.numpy()
                testacc += np.sum(np.round(y_pred[:, 0].data.numpy()) == y_batch.float()[:, 0].data.numpy())
            elapsed_time = time.time() - start_time
            
            if (e % 500 == 0):  
                print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(e + 1, epochs, avg_loss, avg_val_loss, elapsed_time))
            
            testloss.append(avg_val_loss)
            testaccuracy.append(testacc/ len(val_y))        
        plt.title("plot of train,val loss and val accuracy for lr = {}".format(lr))
        plt.plot(trainloss)
        plt.plot(testloss)
        plt.plot(testaccuracy)   
        plt.show()
    return model

'''
This method tests the data and outputs accuracy of the model
'''
def testing(model, test_data, test_labels):
    test_normalized_X = preprocessing.normalize(test_data.values[:,1:])
    test_X = Variable(torch.from_numpy(test_normalized_X)).float()
    
    model.eval()
    y_pred = model(test_X.float()).detach()
    predictions = np.round(y_pred[:, 0].data.numpy())
    print("Accuracy is {}".format(np.round(np.sum(predictions == test_labels.values) / len(test_X), 2)))
    return predictions
      

'''
This method divides the training set into train and validation datasets. 
The splitting is done based on stratified K fold cross validation with k = 3
For every fold, the taining and validation is performed by calling train_pred(model, x_train, y_train, x_val, y_val, epochs) method
Testing function is called and the value of test accuray is obtained
'''
def TrainingAndCV(train_data, train_labels, test_data, test_labels, n_splits = 2):           
    train_normalized_X = preprocessing.normalize(train_data.values[:,1:])
    train_X = Variable(torch.from_numpy(train_normalized_X)).float()
    train_y = torch.from_numpy(train_labels.values.reshape((train_labels.shape[0],-1))).float()
    
    splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7).split(train_X, train_y))
        
    for i, (train_idx, valid_idx) in enumerate(splits):    
        print("K -fold Cross validation for set k = ", i)
        x_train = train_X[train_idx].clone()
        y_train = train_y[train_idx].clone()
        x_val = train_X[valid_idx].clone()
        y_val = train_y[valid_idx].clone()
        model = Feedforward(x_train.shape[1], 20) 
        model = train_pred(model, x_train, y_train, x_val, y_val, epochs=2000)
    testing(model, test_data, test_labels)
    
def main():
    os.chdir("D:\\Spring 2020\\assignments\\senticlassFFNL-master\\senticlassFFNL")
    # print command line arguments
    train= get_data("tweet\\train") #get_data(sys.argv[1])
    test= get_data("tweet\\test")  #get_data(sys.argv[2])
    
    print("Cleaning data...")
    clean_train_stem,clean_train_nostem= clean(train)
    clean_test_stem, clean_test_nostem= clean(test)
    print("Data Cleaning done")
    
    print('shape of stem', clean_train_stem.shape)
    print('shape of non stem', clean_train_nostem.shape)
        
    print("creating Vector representations(TF-IDF)...")
    traindf_stem_tfidf, stem_vocab_tf, trainvector_stem_tfidf, ts_vectorizer= getrep(clean_train_stem, 'tfidf')
    traindf_nostem_tfidf, nostem_vocab_tf, trainvector_nostem_tfidf, tn_vectorizer= getrep(clean_train_nostem, 'tfidf')

    testdf_stem_tfidf = transformtest(traindf_stem_tfidf, clean_test_stem, ts_vectorizer,stem_vocab_tf)
    test_nostem_tfidf = transformtest(traindf_nostem_tfidf, clean_test_nostem, tn_vectorizer, nostem_vocab_tf)
    print("Vector representations Created")
    
    print("Training phase...")
    print("------------------------------------- \n For stem dataset \n-------------------------------------")
    TrainingAndCV(traindf_stem_tfidf, train.Sentiment, testdf_stem_tfidf, test.Sentiment, n_splits = 5)
    print("------------------------------------- \n For non stem dataset \n-------------------------------------")
    TrainingAndCV(traindf_nostem_tfidf, train.Sentiment, test_nostem_tfidf, test.Sentiment, n_splits = 5)
   
class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        nn.init.xavier_normal_(self.fc1.weight)
        self.fc2 = torch.nn.Linear(self.hidden_size, hidden_size)
        nn.init.xavier_normal_(self.fc2.weight)
        self.fc3 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        hidden1 = self.sigmoid(self.fc1(x))
        output = self.sigmoid(self.fc2(hidden1))
        output = self.sigmoid(self.fc3(output))
        return output


if __name__ == "__main__":
    main()








