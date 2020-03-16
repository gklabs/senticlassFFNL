'''
How to run:
File name + location of  data (tweet folder)
e.g. 
python3 Neurallanguagemodel.py /Users/gkbytes/nlm/tweet/train /Users/gkbytes/nlm/tweet/test
########################################
1. get data
	train
	validation
	test
2. clean
	Remove HTML tags
	Markup tags
	Lower case cap letters except stuff like USA
	No stop word removal
	tokenize at white space
	emoticon tokenizer (Tweet tokenizer)
3. bi-gram representation function for train and test
	2 negative sample for each positive sample
	for negative sample, 
		create the vocabulary of training sample
		randomly pick a word other than the word in the positive sample.
4. Create Language Model
	Embedding vector
	One hot encoding of every word in the vocabulary using sklearn
	Initialize Embedding vector of dimensions (d,|V|). We assume d=10 with random numbers
	Feed forward Neural Network 2 Hidden Layers of size 20
	initialize weights with random numbers
	LR= 0.00001/ tune
5. Predict and print accuracy for positive tweets in test
References:
https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
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
import sys
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from nltk.stem.snowball import SnowballStemmer
import torch
import torch.optim as optim
from torch.autograd import Variable
from sklearn import preprocessing
import time
from sklearn.model_selection import StratifiedKFold
nl.download('punkt')


start_time = time.time()

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
	print("words in order len",len(wordsinorder))
	bigrams=[]
	bigramlist=[]
	for i in range(len(wordsinorder)-1):
		bigram_list=[wordsinorder[i], wordsinorder[i + 1], 1]
		#print(bigram_list)
		bigrams.append(bigram_list)
	#print(bigrams)
	return bigrams


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


# returns a list that contains k negative sample bigrams for every given positive bigram
def neg_sample_bigrammer(bigramlist,vocab,k):
	end= len(vocab)
	negsample=[]
	negbigsample= []
	print(bigramlist[1])
	for posbigram in bigramlist:
		word=str(posbigram[1])
		if word in vocab:
			num= vocab.index(word)
			neglist= random.sample([i for i in range(0,end) if i not in [num]],k)
			for j in neglist:
				negsample= [posbigram[0][0], vocab[j], 0]
				negbigsample.append(negsample)
	print(len(negbigsample))
	return negbigsample


class SkipGram(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embeddings_u = torch.nn.Embedding(vocab_size, embedding_dim)
        self.embeddings_v = torch.nn.Embedding(vocab_size, embedding_dim)

    def forward(self, word, context) -> torch.Tensor:
        emb_word = self.embeddings_u(word).view((1, -1))
        emb_cont = self.embeddings_v(context).view((1, -1))
        score = torch.mm(emb_word, torch.t(emb_cont))
        output = torch.nn.functional.logsigmoid(score)
        return output
    
def Languagemodel(bigrams_data, vocab2,d):
    bigram= bigrams_data
    print(bigram[1])
    vocab = set(vocab2)
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    losses = []
    loss_function = torch.nn.MSELoss()
    model = SkipGram(len(vocab), d)
    optimizer = optim.SGD(model.parameters(), lr=0.0001)
    for epoch in range(200):
        total_loss = 0
        for context, target, label in bigram:
           # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
           # into integer indices and wrap them in tensors)
            if context in vocab2:
                context_idxs = torch.tensor(word_to_ix[context], dtype=torch.long) 
                target_idxs = torch.tensor(word_to_ix[target], dtype=torch.long) 
                
                #step 2. Recall that torch *accumulates* gradients. Before passing in a
                # new instance, you need to zero out the gradients from the old instance
                model.zero_grad()
                
                # Step 3. Run the forward pass, getting log probabilities over next words
                log_probs = model(context_idxs, target_idxs)
                
                # Step 4. Compute your loss function. (Again, Torch wants the target
                # word wrapped in a tensor)
                loss = loss_function(log_probs[0], torch.tensor([label], dtype=torch.float))
                
                # Step 5. Do the backward pass and update the gradient
                loss.backward()
                optimizer.step()
                
                # Get the Python number from a 1-element Tensor by calling tensor.item()
                total_loss += loss.item()
        losses.append(total_loss)
        plt.title("plot of training loss ")
        plt.plot(losses)


def Testing(model, vocab, data):
    predictions = []
    actuals = []
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    for context, target, label in data:
        context_idxs = torch.tensor(word_to_ix[context], dtype=torch.long) 
        target_idxs = torch.tensor(word_to_ix[target], dtype=torch.long) 
        model.eval()
        log_probs = model(context_idxs, target_idxs)
        _, prediction = torch.max(log_probs.data, 1)
        prediction = prediction.data[0]
        predictions.append(prediction)
        actuals.append(label)
    print(confusion_matrix(actuals, predictions))
    


def main():
    import os
    os.chdir("D:\\Spring 2020\\assignments\\senticlassFFNL-master\\senticlassFFNL")
    # print command line arguments
    train= get_data("tweet\\train") #get_data(sys.argv[1])
    test= get_data("tweet\\test")  #get_data(sys.argv[2])
    
    print("cleaning data")
    clean_train_stem,clean_train_nostem= clean(train)
    clean_test_stem, clean_test_nostem= clean(test)
    print("cleaning done")
    

    print("creating the vocabulary for stemmed and unstemmed data")
    Vocab_stem = createvocab(clean_train_stem)
    Vocab_nostem = createvocab(clean_train_nostem)
    print("vocabulary created")
    print("Stemmed vocabulary length=",len(Vocab_stem))
    print("No stem vocabulary length=",len(Vocab_nostem))
    
    print("*********TRAINING************")
    print("creating positive bigrams")
    
    train_stem_pos_bigram= pos_sample_bigrammer(clean_train_stem)
    train_nostem_pos_bigram= pos_sample_bigrammer(clean_train_nostem)
    
    print("positive samples for training created")
    print("No of stem pos bigrams=" ,len(train_stem_pos_bigram))
    print("No of no stem pos bigrams=", len(train_nostem_pos_bigram))
    

    print("creating negative sample bigrams")
    train_stem_neg_bigram = neg_sample_bigrammer(train_stem_pos_bigram, Vocab_stem,2)
    train_nostem_neg_bigram = neg_sample_bigrammer(train_nostem_pos_bigram, Vocab_nostem,2)
    print("negative samples created")
    
    print("No of stem neg bigrams=" ,len(train_stem_neg_bigram))
    print("No of no stem neg bigrams=", len(train_nostem_neg_bigram))
 

    #create a training dataframe with positive and negative samples and adding labels(1,0)  for them
    train_stem_data = train_stem_pos_bigram + train_stem_neg_bigram
    train_nostem_data= train_nostem_pos_bigram + train_nostem_neg_bigram
    
    #nostem_labels =[0]*len(train_nostem_pos_bigram) + [1]*len(train_nostem_neg_bigram)
    #stem_labels= [0]*len(train_stem_pos_bigram)+ [1]*len(train_stem_neg_bigram)

    print("--- %s seconds ---" %(time.time() - start_time))
    print("train data is ready for stem and no stem ")

    print("Creating Language model")
    stem_model =  Languagemodel(train_stem_data, Vocab_stem, 10)
    Languagemodel(train_nostem_data, Vocab_stem, 10)
    
    print("*********TESTING************")
    print("creating positive bigrams")
    
    test_stem_pos_bigram= pos_sample_bigrammer(clean_test_stem)
    test_nostem_pos_bigram= pos_sample_bigrammer(clean_test_nostem)
    
    print("positive samples for testing created")
    print("No of no stem pos bigrams=", len(test_stem_pos_bigram))
    print("No of stem pos bigrams=" ,len(test_nostem_pos_bigram))
    
    print("creating the vocabulary for stemmed and unstemmed Test data")
    Vocab_stem_test = createvocab(clean_test_stem)
    Vocab_nostem_test = createvocab(clean_test_nostem)
    
    print("creating negative sample bigrams")
    test_stem_neg_bigram = neg_sample_bigrammer(test_stem_pos_bigram, Vocab_stem_test,2)
    test_nostem_neg_bigram = neg_sample_bigrammer(test_nostem_pos_bigram, Vocab_nostem_test,2)
    print("negative samples created")
    
    print("No of no stem neg bigrams=", len(test_nostem_neg_bigram))
    print("No of stem neg bigrams=" ,len(test_stem_neg_bigram))

    #Create labels
    test_stem_data = test_stem_pos_bigram + test_stem_neg_bigram
    test_nostem_data= test_nostem_pos_bigram + test_nostem_neg_bigram
    
    #Testing
    Testing(stem_model, Vocab_stem_test, test_stem_data)
    Testing(stem_model, Vocab_nostem_test, test_nostem_data)
    
if __name__ == "__main__":
    main()
