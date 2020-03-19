'''
Author: Giridhar R, Haritha G
Description: AIT 726 Homework 2 - Part2
Command to run the file:
python LanguageModel.py [train folder location] [test folder location] 

Description: Language modeling as a binary classification of positive and negative Bi-grams

Detailed Procedure:
1. import dependencies
2. get data
    Read training data of positive tweets
    Read testing data of positive tweets
3. Clean data
    remove html tags, flight names starting with @, links (strings with //)
    lower case the word if the first letter is uppercase
    Stop words are not removed
    Tokenize the word and emojis using TweetTokenizer on nltk.tokenize.casual 
    Create seperate datasets with stemmed tokens and non stemmed tokens 
4. bi-gram representation function for train and test
	2 negative sample for each positive sample
	for negative sample, 
		create the vocabulary of training sample
		randomly pick a word other than the word in the positive sample.
5. Create Bigram Language Model using FFNN
	5.1. Obtain the vector reprresentations of train set using TF-IDF algorithm
    5.2. Model building using Feedforward in Pytorch's nn.Module      
         dimensions - Input layer : size of vocab(2651 for stem, 3215 for non-stem)
                   Hidden layers: 20 nodes 
                   Output layer: 1 which is the probability score if the bigram is likely to appear in the language model      
      Sigmoid as Activation function for the linear layers
      Initialize weights to random numbers
    5.3. Training
      Train the model on the vectorized representation of train data
      Use k-fold cross validation for k=4 to test the generalizability of the model
      Run the training for 2000 epochs and learning rates set = [0.0001, 0.05, 0.1, 0.5, 1] 
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
   Create positive and negative bigram samples similar to training part
   Vector represenation of test:
       Create Term frequncy matrix for all the tokens in test data
       Multipy TF of test data with the IDF representation of train data to get test vector representation
   Use tuned parameters of the final neural network 
   Compute the probabilities for the test
   Print Accuracy
   Print Confusion matrix
	
References:
https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html

###########Results###############
  For stem: Accuracy is 0.9
            Confusion matrix  [[32459  3925]
                              [ 1707 16485]]
      
  For non-stem: Accuracy is 0.88
                Confusion matrix [[31735  4635]
                                 [ 1698 16487]]
	 
Detailed run results of the program along with the metrics for training loss, validation loss and validation accuracy 
are attched as LanguageModelOutput.log file 

'''

# import dependencies
import pandas as pd
from collections import defaultdict
from pathlib import Path
import nltk as nl
from nltk.tokenize import word_tokenize
import re
from nltk.tokenize.casual import TweetTokenizer
import numpy as np
import sys, math
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from nltk.stem.snowball import SnowballStemmer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from sklearn import preprocessing
import time
from sklearn.model_selection import StratifiedKFold
nl.download('punkt')
import random

start_time = time.time()
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
    df_pos['Sentiment']=1    
    return df_pos

'''
This method takes care of cleaning the data
    remove html tags, flight names starting with @, links (strings with //)
    lower case the word if the first letter is uppercase
    remove stop words
    Tokenize the word and emojis using TweetTokenizer on nltk.tokenize.casual 
    Create seperate datasets with stemmed tokens and non stemmed tokens 
'''
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
   
    return df_stemmed,df_unstemmed


def allwordsinorder(df):
	allwords= []
	for text in df.text:
		for word in text:
			allwords.append(word)
	return allwords

#returns a list containing bigrams
def pos_sample_bigrammer(df):
	wordsinorder= allwordsinorder(df)
	print("\t words in order len",len(wordsinorder))
	bigrams=[]
	for i in range(len(wordsinorder)-1):
		bigram_list=[wordsinorder[i], wordsinorder[i + 1]]
		#print(bigram_list)
		bigrams.append(bigram_list)
	#print(bigrams)
	return bigrams

# returns a list that contains k negative sample bigrams for every given positive bigram
def neg_sample_bigrammer(bigramlist,vocab,k):
	end= len(vocab)
	negsample=[]
	negbigsample= []
	for posbigram in bigramlist:
		word=str(posbigram[1])
		if word in vocab:
			num= vocab.index(word)
			neglist= random.sample([i for i in range(0,end) if i not in [num]],k)
			for j in neglist:
				negsample= [posbigram[0][0], vocab[j]]
				negbigsample.append(negsample)
	return negbigsample

#creating vocabulary
def createvocab(df):
    V=[]
    for tweet in df.text:
        for keyword in tweet:
            if keyword  in V:
                continue
            else :
                V.append(keyword)
    return V

'''
This method is used to create vector representation for the tokenized data.
Steps:Calculate Term Frequency of every token in every tweet
      Calculate Document Frequency of every token across training data
      Compute Inverse Document Frequency (formula)
      Compute TF*IDF
      Create a normalized Matrix representation of TF*IDF (embedding) and return it.
'''
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

        return TFIDF_dataframe, Vocab, invdocufreq

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
        
        #Convert to tensor data
        train_tsdata = torch.utils.data.TensorDataset(train_X, train_y)
        valid_tsdata = torch.utils.data.TensorDataset(val_X, val_y)
        
        #Feed the tensor data to data loader. This partitions the data based on batch size
        train_loader = torch.utils.data.DataLoader(train_tsdata, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_tsdata, batch_size=batch_size, shuffle=False)
        
        #Run the model on train set and capture loss
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
        
        #Run the model on validation set and capture loss
        model.eval()          
        avg_val_loss = 0.
        testacc = 0
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            X_val = Variable(torch.FloatTensor(x_batch))
            y_pred_val = model(X_val)
            avg_val_loss += loss_fn(y_pred_val, y_batch.float()).item() / len(valid_loader)
            valid_preds[i * batch_size:(i+1) * batch_size] = y_pred_val[:, 0].data.numpy()
            testacc += np.sum(np.round(y_pred_val[:, 0].data.numpy()) == y_batch.float()[:, 0].data.numpy())
        elapsed_time = time.time() - start_time
        
        if (e % 100 == 99):  
            print('\t Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(e + 1, epochs, avg_loss, avg_val_loss, elapsed_time))
       
        testloss.append(avg_val_loss)
        testaccuracy.append(testacc/ len(val_y))   
    #Visualize the trainloss, validation loss and validation accuracy
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
        optimizer = optim.SGD(model.parameters(), lr=lr)  #Optimizing with Stochastic Gradient Descent
        loss_fn = nn.MSELoss()  # Mean Squared Error Loss
       
        train_tsdata = torch.utils.data.TensorDataset(train_X, train_y)         #Convert to tensordata
        train_loader = torch.utils.data.DataLoader(train_tsdata, batch_size=batch_size, shuffle=True) #Dataloader using batch size
        
        #Run the model on entire train set 
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
    return model

'''
This method tests the data and outputs accuracy of the model
'''
def testing(model, test_data, test_labels):
    test_normalized_X = preprocessing.normalize(test_data.values[:,1:])
    test_X = Variable(torch.from_numpy(test_normalized_X)).float()
    #Evaluate on test data
    model.eval()
    y_pred = model(test_X.float()) #Get predictions
    predictions = np.round(y_pred[:, 0].data.numpy()) 
    #compute accuracy and confusion matrix
    cm = confusion_matrix(test_labels.values, predictions, labels=None)
    print("\t Confusion matrix \n \t {}".format(cm))
    print("\t Accuracy is {}".format(np.round(np.sum(predictions == test_labels.values) / len(test_X), 2)))
    return predictions   

'''
Define subclass that inherits from super class Feedforward in nn.Module
'''   
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

'''
This method divides the training set into train and validation datasets. 
The splitting is done based on stratified K fold cross validation with k = 4
For every fold, the taining and validation is performed by calling train_pred(model, x_train, y_train, x_val, y_val, epochs) method
Testing function is called and the value of test accuray is obtained
'''
def TrainingAndCV(train_data, train_labels, n_splits = 2):           
    train_normalized_X = preprocessing.normalize(train_data.values[:,1:])
    train_X = torch.from_numpy(train_normalized_X).float()
    train_y = torch.from_numpy(train_labels.values.reshape((train_labels.shape[0],-1))).float()
    
    #Initialize K fold cross validation 
    splits = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7).split(train_X, train_y))
    print("\t 4.1. K-fold cross-validation - Each k-th fold runs for different learning rates")
    lrs = [0.0001, 0.05, 0.1, 0.5, 1]
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
            lrdict[lr] = train_pred_ModelTuning(model, x_train, y_train, x_val, y_val, lr, epochs=20, batch_size=400)
    #Obtain the learning rate corresponding to minimum loss and train the data 
    tunedlr = min(lrdict, key=lrdict.get)
    print("\t On cross-validation the best parameter for learning rate was found to be ", tunedlr)
    print("\t 4.2. Run the training with entire train data and learning rate = ", tunedlr)
    #Final model training step after parameter tuning and made sure that model doesnot overfit
    model = training(train_X, train_y, lr= tunedlr, epochs=500, batch_size=400)
    return model

#Main method
def main():
    print("1. Reading data...")
    train= get_data(sys.argv[1])
    test= get_data(sys.argv[2])
    
    print("2. Cleaning data...")
    clean_train_stem,clean_train_nostem= clean(train)
    
    clean_test_stem, clean_test_nostem= clean(test)
    print("\t cleaning done")
    

    print("3. creating the vocabulary for stemmed and unstemmed data")
    Vocab_stem = createvocab(clean_train_stem)
    Vocab_nostem = createvocab(clean_train_nostem)
    print("\t vocabulary created")
    print("\t Stemmed vocabulary length=",len(Vocab_stem))
    print("\t No stem vocabulary length=",len(Vocab_nostem))
    
    print("*********TRAINING************")
    print("4. creating positive bigrams")    
    train_stem_pos_bigram= pos_sample_bigrammer(clean_train_stem)
    train_nostem_pos_bigram= pos_sample_bigrammer(clean_train_nostem)
    
    print("\t positive samples for training created")
    print("\t No of stem pos bigrams=" ,len(train_stem_pos_bigram))
    print("\t No of no stem pos bigrams=", len(train_nostem_pos_bigram))
    print("\t few samples from stemmed: \n \t :", train_stem_pos_bigram[:5])

    print("5. creating negative sample bigrams")
    train_stem_neg_bigram = neg_sample_bigrammer(train_stem_pos_bigram, Vocab_stem,2)
    train_nostem_neg_bigram = neg_sample_bigrammer(train_nostem_pos_bigram, Vocab_nostem,2)
    print("\t negative samples created")
    
    print("\t No of stem neg bigrams=" ,len(train_stem_neg_bigram))
    print("\t No of no stem neg bigrams=", len(train_nostem_neg_bigram))
    print("\t few samples from stemmed: \n \t :", train_stem_neg_bigram[:5])

    #create a training dataframe with positive and negative samples and adding labels(1,0)  for them
    train_stem_data = train_stem_pos_bigram + train_stem_neg_bigram
    train_nostem_data= train_nostem_pos_bigram + train_nostem_neg_bigram
    
    stem_labels= [1]*len(train_stem_pos_bigram)+ [0]*len(train_stem_neg_bigram)
    nostem_labels =[1]*len(train_nostem_pos_bigram) + [0]*len(train_nostem_neg_bigram)    

    traindf_stem_data = pd.DataFrame()
    traindf_stem_data['text'] = pd.Series((i for i in train_stem_data))
    traindf_stem_data['Sentiment'] = stem_labels
    traindf_nostem_data = pd.DataFrame()
    traindf_nostem_data['text'] = pd.Series((i for i in train_nostem_data))
    traindf_nostem_data['Sentiment'] = nostem_labels
    
    #creating Vector representations(TF-IDF)
    print("6. creating Vector representations(TF-IDF)")
    traindf_stem_tfidf, stem_vocab_tf, ts_vectorizer= getrep(traindf_stem_data, 'tfidf')
    traindf_nostem_tfidf, nostem_vocab_tf, tn_vectorizer= getrep(traindf_nostem_data, 'tfidf')
    print("\t train data is ready for stem and no stem ")
    
    #Training the vectorized bigrams using Binary classication FFNN
    print("7. Training the vectorized bigrams using Binary classication FFNN ")
    print("\t ------------------------------------- \n \t For stem dataset \n \t -------------------------------------")
    model_stem = TrainingAndCV(traindf_stem_tfidf, traindf_stem_data.Sentiment, n_splits = 3)
    print("\t ------------------------------------- \n \t For non stem dataset \n \t -------------------------------------")
    model_nostem = TrainingAndCV(traindf_nostem_tfidf, traindf_nostem_data.Sentiment, n_splits = 3)
    print("\t training done for stem and no stem ")
    
    #Testing phase
    print("*********TESTING************")
    print("8. creating positive bigrams")    
    test_stem_pos_bigram= pos_sample_bigrammer(clean_test_stem)
    test_nostem_pos_bigram= pos_sample_bigrammer(clean_test_nostem)
    
    print("\t positive samples for testing created")
    print("\t No of no stem pos bigrams=", len(test_stem_pos_bigram))
    print("\t No of stem pos bigrams=" ,len(test_nostem_pos_bigram))
    
    print("9. creating the vocabulary for stemmed and unstemmed Test data")
    Vocab_stem_test = createvocab(clean_test_stem)
    Vocab_nostem_test = createvocab(clean_test_nostem)
    
    print("10. creating negative sample bigrams")
    test_stem_neg_bigram = neg_sample_bigrammer(test_stem_pos_bigram, Vocab_stem_test,2)
    test_nostem_neg_bigram = neg_sample_bigrammer(test_nostem_pos_bigram, Vocab_nostem_test,2)
    print("\t negative samples created")
    
    print("\t No of no stem neg bigrams=", len(test_nostem_neg_bigram))
    print("\t No of stem neg bigrams=" ,len(test_stem_neg_bigram))
    
    #Combine test and train into one dataset
    test_stem_data = test_stem_pos_bigram + test_stem_neg_bigram
    test_nostem_data= test_nostem_pos_bigram + test_nostem_neg_bigram
    #Create labels
    stem_labels= [1]*len(test_stem_pos_bigram)+ [0]*len(test_stem_neg_bigram)
    nostem_labels =[1]*len(test_nostem_pos_bigram) + [0]*len(test_nostem_neg_bigram)    
    
    testdf_stem_data = pd.DataFrame()
    testdf_stem_data['text'] = pd.Series((i for i in test_stem_data))
    testdf_stem_data['Sentiment'] = stem_labels
    testdf_nostem_data = pd.DataFrame()
    testdf_nostem_data['text'] = pd.Series((i for i in test_nostem_data))
    testdf_nostem_data['Sentiment'] = nostem_labels
    
    print("11. Vector representations for test stem and non-stem created")
    testdf_stem_tfidf = transformtest(traindf_stem_tfidf, testdf_stem_data, ts_vectorizer,stem_vocab_tf)
    test_nostem_tfidf = transformtest(traindf_nostem_tfidf, testdf_nostem_data, tn_vectorizer, nostem_vocab_tf)
    
    #Testing the model
    print("12. Testing the model")
    print("\t ------------------------------------- \n \t For stem dataset \n \t -------------------------------------")
    testing(model_stem, testdf_stem_tfidf, testdf_stem_data.Sentiment)
    print("\t ------------------------------------- \n \t For non stem dataset \n \t -------------------------------------")
    testing(model_nostem, test_nostem_tfidf, testdf_nostem_data.Sentiment)
    
if __name__ == "__main__":
    main()
