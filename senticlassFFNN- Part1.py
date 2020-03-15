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
4. TF IDF represenattion from scratch
      TF- IDF representation from scratch
      Calculate Term Frequency of every token in every tweet
      Calculate Document Frequency of every token across training data
      Compute Inverse Document Frequency (formula)
      Compute TF*IDF
      Create a normalized Matrix representation of TF*IDF (embedding) and return it.  
5.Build FFNN 
part1: Model building using Feedforward in Pytorch's nn.Module      
      dimensions - Input layer : size of vocab(5639 for stem, 7269 for unstemmed)
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
          Compute validation accuracy
      Visualize the loss for each epoch and validation accuracy for each epoch
7. Predict on Test set
   Clean Test Data
   Vector represenation of test:
       Create Term frequncy matrix for all the tokens in test data
       Multipy TF of test data with the IDF representation of train data to get test vector representation
   Use tuned parameters of the final neural network 
   Compute the probabilities for the test
   Print Accuracy
   Print Confusion matrix
            
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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn import preprocessing
import time
from sklearn.model_selection import StratifiedKFold
nl.download('punkt')
nl.download('stopwords')

# pass path as train and test folder

stdoutorigin = sys.stdout
sys.stdout = open("D:\\Spring 2020\\assignments\\senticlassFFNL-master\\log_nonormalizedtfidf.txt", "w")
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

    return df

def clean(df):
    # Remove any markup tags (HTML), all the mentions of handles(starts with '@') and '#' character
    def cleantweettext(raw_html):
        pattern = re.compile('<.*?>')
        cleantext = re.sub(pattern, '', raw_html)
        cleantext = " ".join(filter(lambda x:x[0]!='@', cleantext.split()))
        cleantext = cleantext.replace('#', '')
        return cleantext
    #Remove words start with @(handles)
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
    
    #Uncapitalize words with first letter as uppercase
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
    
    #Do not include stop words
    def removedstopwords(tweettokens):
        return [word for word in tweettokens if word not in stopwords.words('english')]
    
    #remove the urls. This step is done as they are scored high by TFIDF due to their infrequency
    def removeUrls(tweettokens):
        newtweettokentext = []
        for token in tweettokens:
            if (not re.compile(r'^//t.co/').search(token)) and (token != 'http' and token != 'https'):
                newtweettokentext.append(token)
        return newtweettokentext
    
    #Calling athe above functions on the data
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
   
    nostopwordtweet=[]
    for doc in df.text:
        nostopwordtweet.append(removeUrls(removedstopwords(doc)))
    df.text =nostopwordtweet
       
    #perform stemming using SnowballStemmer
    stemtweets=[]
    stemmer = SnowballStemmer("english", ignore_stopwords=False)
    
    for x in df.text:
        stemtweet=''
        for word in x:
            stemtweet=stemtweet+stemmer.stem(word)+' '
        stemtweets.append(word_tokenize(stemtweet))
    df['stemmed']=stemtweets

    #Craete seperate dataframes for stemmed and unstemmed tokenized tweets and return
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
      Create a normalized Matrix representation of TF*IDF (embedding) and return it.
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
        
        #Compute document frequency
        for word in Vocab:
            for tweet in df.text:
                if wordintweet(tweet, word) == 1:
                    docufreq[word]+=1
        
        #Compute inverse - document frequency
        for word in docufreq:
            invdocufreq[word]= math.log(N/docufreq[word],10)
        
        TFIDF_dataframe = pd.DataFrame(0,index= list(set(freqdict.keys())),columns= Vocab)
        
        #Multiply tf and Idf 
        for wid,info in freqdict.items():
            for keyword in info:
                if info[keyword] > 0:
                    tfscore = 1+math.log(info[keyword],10)
                else:
                    tfscore = 0
                TFIDF_dataframe.at[wid,keyword]= tfscore * invdocufreq[keyword]

        TFIDFmatrix= TFIDF_dataframe.values
        return TFIDF_dataframe,Vocab,TFIDFmatrix, invdocufreq

'''
This method creates vector representation for test data:
Create Term frequncy matrix for all the tokens in test data
Multipy TF of test data with the IDF representation of train data to get test vector representation
'''  
def transformtest(refDF, testdf, invdocufreq, trainVocab):    
    freqdict={}
    #compute term frequency for test
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
def train_pred_ModelTuning(model, train_X, train_y, val_X, val_y, lr, epochs=2, batch_size=200):    
    valid_preds = np.zeros((val_X.size(0)))     
    trainloss = []
    testloss = []
    testaccuracy = []
              
    for e in range(epochs):
        start_time = time.time()
                       
        optimizer = optim.SGD(model.parameters(), lr=lr)  #Optimizing with Stochastic Gradient Descent
        loss_fn = nn.MSELoss()  # Mean Squared Error Loss
       
        train_tsdata = torch.utils.data.TensorDataset(train_X, train_y)
        valid_tsdata = torch.utils.data.TensorDataset(val_X, val_y)
        
        train_loader = torch.utils.data.DataLoader(train_tsdata, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_tsdata, batch_size=batch_size, shuffle=False)
      
        model.train()
        avg_loss = 0.
        for x_batch, y_batch in train_loader:
            X = Variable(torch.FloatTensor(x_batch))
            y = Variable(torch.FloatTensor(y_batch))
            optimizer.zero_grad() #null the gradients
            y_pred = model(X) #forward pass
            loss = loss_fn(y_pred.squeeze(), y.squeeze()) #Compute loss
            loss.backward() #back propagate
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
        trainloss.append(avg_loss)
        
        model.eval()          
        avg_val_loss = 0.
        testacc = 0
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            X_val = Variable(torch.FloatTensor(x_batch))
            y_pred_val = model(X_val)
            avg_val_loss += loss_fn(y_pred_val, y_batch.float()).item() / len(valid_loader)
            valid_preds[i * batch_size:(i+1) * batch_size] = y_pred_val[:, 0].data.numpy()
            testacc += np.sum(np.round(y_pred_val[:, 0].data.numpy()) == y_batch.float()[:, 0].data.numpy())
            #if (e % 10 == 0):
                #print("predictions ......", y_pred_val[:, 0].data.numpy())
                #print("Actuals.......", y_batch.float()[:, 0].data.numpy())
        elapsed_time = time.time() - start_time
        
        if (e % 500 == 0):  
            print('\t Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(e + 1, epochs, avg_loss, avg_val_loss, elapsed_time))
       
        testloss.append(avg_val_loss)
        testaccuracy.append(testacc/ len(val_y))        
    plt.title("plot of train,val loss and val accuracy for lr = {}".format(lr))
    plt.plot(trainloss)
    plt.plot(testloss)
    plt.plot(testaccuracy)   
    plt.show()
    return min(testloss)

'''
This method is called for final training on the entire train data with the tuned parameters
'''
def training(train_X, train_y, lr, epochs=2, batch_size=200):   
    model = Feedforward(train_X.shape[1], 20)  
    print("\t Model Summary:")
    print("\t ", model)      
    trainloss = []              
    for e in range(epochs):
        start_time = time.time()
                       
        optimizer = optim.SGD(model.parameters(), lr=lr)  #Optimizing with Stochastic Gradient Descent
        loss_fn = nn.MSELoss()  # Mean Squared Error Loss
       
        train_tsdata = torch.utils.data.TensorDataset(train_X, train_y)        
        train_loader = torch.utils.data.DataLoader(train_tsdata, batch_size=batch_size, shuffle=True)
      
        model.train()
        avg_loss = 0.
        for x_batch, y_batch in train_loader:
            X = Variable(torch.FloatTensor(x_batch))
            y = Variable(torch.FloatTensor(y_batch))
            optimizer.zero_grad() #null the gradients
            y_pred = model(X) #forward pass
            loss = loss_fn(y_pred.squeeze(), y.squeeze()) #Compute loss            
            loss.backward() #back propagate
            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
        trainloss.append(avg_loss)
        elapsed_time = time.time() - start_time
        if (e % 499 == 0):  
            print('\t Epoch {}/{} \t training loss ={:.4f} \t time={:.2f}s'.format(e + 1, epochs, avg_loss, elapsed_time))
    return model

'''
This method tests the data and outputs accuracy of the model
'''
def testing(model, test_data, test_labels):
    test_normalized_X = preprocessing.normalize(test_data.values[:,1:])
    test_X = Variable(torch.from_numpy(test_normalized_X)).float()
    
    model.eval()
    y_pred = model(test_X.float()) #Get predictions
    predictions = np.round(y_pred[:, 0].data.numpy()) 
    #compute accuracy and confusion matrix
    cm = confusion_matrix(test_labels.values, predictions, labels=None)
    print("\t Confusion matrix \n \t {}".format(cm))
    print("\t Accuracy is {}".format(np.round(np.sum(predictions == test_labels.values) / len(test_X), 2)))
    return predictions  

'''
This method divides the training set into train and validation datasets. 
The splitting is done based on stratified K fold cross validation with k = 3
For every fold, the taining and validation is performed by calling train_pred(model, x_train, y_train, x_val, y_val, epochs) method
Testing function is called and the value of test accuray is obtained
'''
def TrainingAndCV(train_data, train_labels, n_splits = 2):           
    train_normalized_X = train_data.values[:,1:]#preprocessing.normalize()
    train_X = torch.from_numpy(train_normalized_X).float()
    train_y = torch.from_numpy(train_labels.values.reshape((train_labels.shape[0],-1))).float()
    
    #Initialize K fold cross validation 
    splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7).split(train_X, train_y))
    print("\t 4.1. K-fold cross-validation")
    lrs = [0.0001, 0.05, 0.1, 1]
    lrdict= dict((el,0) for el in lrs)
    #Cross validation and parameter tuning
    for i, (train_idx, valid_idx) in enumerate(splits):    
        print("\t K -fold Cross validation for set k = ", i+1)
        x_train = train_X[train_idx].clone()
        y_train = train_y[train_idx].clone()
        x_val = train_X[valid_idx].clone()
        y_val = train_y[valid_idx].clone()
        model = Feedforward(x_train.shape[1], 20)        
        for lr in lrs:
            lrdict[lr] = train_pred_ModelTuning(model, x_train, y_train, x_val, y_val, lr, epochs=100, batch_size=200)
    #Obtain the learning rate corresponding to minimum loss and train the data 
    tunedlr = min(lrdict, key=lrdict.get)
    print("\t On cross-validation the best parameter for learning rate was found to be ", tunedlr)
    print("\t 4.2. Run the training with entire train data and learning rate = ", tunedlr)
    model = training(train_X, train_y, lr= tunedlr, epochs=2000, batch_size=200)
    return model
    
def main():
    print("1. Reading data...")
    os.chdir("D:\\Spring 2020\\assignments\\senticlassFFNL-master\\senticlassFFNL")
    # print command line arguments
    train= get_data("tweet\\train") #get_data(sys.argv[1])
    test= get_data("tweet\\test")  #get_data(sys.argv[2])
    
    print("2. Cleaning data...")
    clean_train_stem,clean_train_nostem= clean(train)    
    print("\t Data Cleaning done")
    
    print('shape of stem', clean_train_stem.shape)
    print('shape of non stem', clean_train_nostem.shape)
        
    print("3. creating Vector representations(TF-IDF)...")
    traindf_stem_tfidf, stem_vocab_tf, trainvector_stem_tfidf, ts_vectorizer= getrep(clean_train_stem, 'tfidf')
    traindf_nostem_tfidf, nostem_vocab_tf, trainvector_nostem_tfidf, tn_vectorizer= getrep(clean_train_nostem, 'tfidf')
    print("\t Vector representations Created")
    
    print("4. Training the model...")
    print("\t ------------------------------------- \n \t For stem dataset \n \t -------------------------------------")
    model_stem = TrainingAndCV(traindf_stem_tfidf, train.Sentiment, n_splits = 5)
    print("\t ------------------------------------- \n \t For non stem dataset \n \t -------------------------------------")
    model_nostem = TrainingAndCV(traindf_nostem_tfidf, train.Sentiment, n_splits = 5)
    
    print("5. Preprocess testing data...")
    clean_test_stem, clean_test_nostem= clean(test)
    print("\t test data cleaned")    
    testdf_stem_tfidf = transformtest(traindf_stem_tfidf, clean_test_stem, ts_vectorizer,stem_vocab_tf)
    test_nostem_tfidf = transformtest(traindf_nostem_tfidf, clean_test_nostem, tn_vectorizer, nostem_vocab_tf)
    print("\t Vector representations for test stem and non-stem created")
    print("6. Testing the model...")
    print("\t ------------------------------------- \n \t For stem dataset \n \t -------------------------------------")
    testing(model_stem, testdf_stem_tfidf, test.Sentiment)
    print("\t ------------------------------------- \n \t For non stem dataset \n \t -------------------------------------")
    testing(model_nostem, test_nostem_tfidf, test.Sentiment)
   
class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size) #linear layer(vocab_size, 20)
        nn.init.xavier_uniform_(self.fc1.weight) #Initialize weights to random numbers chosen from uniform dist
        self.fc2 = torch.nn.Linear(self.hidden_size, hidden_size) #Linear layer(20, 20)
        nn.init.xavier_uniform_(self.fc2.weight) #Initialize weights to random numbers chosen from uniform dist
        self.fc3 = torch.nn.Linear(self.hidden_size, 1) #Linear layer(20, 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        hidden1 = self.sigmoid(self.fc1(x)) #Sigmoid(Input layer)
        output = self.sigmoid(self.fc2(hidden1)) #sigmoid(hidden layer)
        output = self.sigmoid(self.fc3(output)) #sigmoid(hidden layer)
        return output


if __name__ == "__main__":
    main()








