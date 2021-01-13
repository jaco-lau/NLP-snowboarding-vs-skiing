#!/usr/bin/env python
# coding: utf-8

import requests
import time
import pandas as pd

from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem.porter import PorterStemmer
from nltk import FreqDist, pos_tag
from nltk.tokenize import RegexpTokenizer

from transformers import pipeline

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix,plot_confusion_matrix



def get_comments(x,subreddit):
    '''
    Pull x comments off two subreddits and return a DataFrame that distinguishes
    between the two subreddits
    
    Arguments:
    --------------
    x : int, number of comments you will pull from each subreddit
    
    subreddit_1 : string, name of the subreddit you wish to pull comments from
    '''
    n = 0
    
    # use the last created_utc to start the next comment pull
    last = ''
    
    # store comments
    comment = []

    # loop until x comments are pulled
    while n < x:
        url = f'http://api.pushshift.io/reddit/comment/search/?subreddit={subreddit}'
        res = requests.get(f'{url}&before={last}')
        json = res.json()
        
        # loop to add comments to comment list and record last created utc
        for i in range(len(json['data'])):
            comment.append(json['data'][i]['body'])
            n += 1
            last = json['data'][i]['created_utc']
        time.sleep(1)
        
    return comment



# Thanks to Prasoon for helping me with my docstring format
# https://www.python.org/dev/peps/pep-0257/

def get_comments_df(x,subreddit_1,subreddit_2):
    '''
    Pull x comments off two subreddits and return a DataFrame that distinguishes
    between the two subreddits
    
    Arguments:
    --------------
    x : int, number of comments you will pull from each subreddit
    
    subreddit_1 : string, name of first subreddit you wish to pull comments from
    
    subreddit_2 : string, name of second subreddit you wish to pull comments from
    '''
    n = 0
    
    # use the last created_utc to start the next comment pull
    last = ''
    
    # store comments
    comment_list_1 = []

    # loop until x comments are pulled
    while n < x:
        url = f'http://api.pushshift.io/reddit/comment/search/?subreddit={subreddit_1}'
        res = requests.get(f'{url}&before={last}')
        json = res.json()
        
        # loop to add comments to comment list and record last created utc
        for i in range(len(json['data'])):
            comment_list_1.append(json['data'][i]['body'])
            n += 1
            last = json['data'][i]['created_utc']
        time.sleep(1)

    # create a dataframe with first subreddits comments
    df_1 = pd.DataFrame(comment_list_1, columns = ['comments'])
    # set to distinguish the first subreddits comments from the second
    df_1['ski or sb'] = 1
    
    
    # rerun process for second subreddit
    count = 0
    last_utc = ''
    comment_list_2 = []

    # loop until x comments are pulled
    while count < x:
        url = f'http://api.pushshift.io/reddit/comment/search/?subreddit={subreddit_2}'
        res = requests.get(f'{url}&before={last}')
        json = res.json()
        
        # loop to add comments to comment list and record last created utc
        for i in range(len(json['data'])):
            comment_list_2.append(json['data'][i]['body'])
            count += 1
            last_utc = json['data'][i]['created_utc']
        time.sleep(1)

    # create a dataframe with first subreddits comments        
    df_2 = pd.DataFrame(comment_list_2,columns = ['comments'])
    # set to distinguish the second subreddits comments from the first    
    df_2['ski or sb'] = 0
    
    # concatinate both dataframes to create a sigle dataframe with all comments
    df_3 = pd.concat([df_1,df_2])
    
    df_3.reset_index(drop = True, inplace = True)
    
    return df_3



def clean_punc(comment):
    '''
    Removes punctuation from a post/comment
    
    Arguments:
    --------------
    df : Dataframe
    
    column : Series
    '''
    # Instantiate Regex to remove all punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    
    # apply Regex to the comment
    comment_tokens = tokenizer.tokenize(comment.lower())
    
    # return the comments in a string
    return ' '.join(comment_tokens)

def puncless_column(df,column):
    '''
    Removes punctuation from posts/comments in a column
    
    Arguments:
    --------------
    df : Dataframe
    
    column : Series
    '''
    # add a row that to removes all punctuation from each comment by applying
    # clean_punc to each row
    df['no punctuation comments'] = df[column].apply(clean_punc)
    
    return df



# thanks to Prasoon I was able to get over the hump of applying the removal of
# stopwords from multiple comments by splitting my function into two functions
def clean_comments(comment):
    '''
    Removes stop words from a post/comment
    
    Arguments:
    --------------
    df : Dataframe
    
    column : Series
    '''
    # Instantiate ski_and_sb_stopword_list
    stopwords = ski_and_sb_stopword_list()

    clean_comments = []

    # loop to creat a list of words in a comment that aren't in the stopword
    # list
    for word in comment.split():
        if word.lower() in stopwords:
            pass
        else:
            clean_comments.append(word)
            
    # turn the comment into a string
    clean_comments_string = ' '.join(clean_comments)
        
    return clean_comments_string

def clean_column(df,column):
    '''
    Removes stop words from posts/comments in a column
    
    Arguments:
    --------------
    df : Dataframe
    
    column : Series
    '''
    # create a column that applies clean_comments to comments in a column
    df['stopwordless comments'] = df[column].apply(clean_comments)
    
    return df



def remove_single_letter_words(comment):
    '''
    Removes single letter words from a post/comment
    
    Arguments:
    --------------
    df : Dataframe
    
    column : Series
    '''
    clean_comments = []

    # loop to create a list that includes only words in a comment that are
    # more than 2 letters 
    for word in comment.split():
        if len(word) == 1:
            pass
        else:
            clean_comments.append(word)
    
    # turns the list into a string        
    clean_comments_string = ' '.join(clean_comments)
        
    return clean_comments_string

def remove_single_letter_column(df,column):
    '''
    Removes single letter words from posts/comments in a column and removes
    rows with no comments
    
    Arguments:
    --------------
    df : Dataframe
    
    column : Series
    '''
    # remove all single letter words from a column
    df[column] = df[column].apply(remove_single_letter_words)

    # create a list of rows that have blank strings in the column
    blank_columns = df[df[column] == ''].index

    # remove all rows with blank strings 
    df.drop(index = blank_columns,inplace = True)

    # reset index after dropping rows
    df.reset_index(drop = True,inplace = True)
    
    return df



def remove_duplicates(df,column):
    '''
    Removes duplicate posts/comments from df
    
    Arguments:
    --------------
    df : Dataframe
    
    column : Series
    '''
    duplicates = []

    # loop to identify duplicate comments
    for x in range(len(df[column])-1):
        if df[column][x] == df[column][x+1]:
            duplicates.append(x)
            
    # remove all rows with duplicates 
    df.drop(index = duplicates,inplace = True)

    # reset index after dropping rows
    return df.reset_index(drop = True,inplace = True)



def p_stem(string):
    '''
    Convert a string into a string with each word stemmed via PorterStemmer
    
    Arguments:
    --------------
    string: String, a string to 
    '''
    # instantiate PorterStemmer
    p_stemmer = PorterStemmer()
    
    # turn comment into a list of words
    words = string.lower().split()
    
    comment = []
    
    # loop to create a list of the stemmed words from the comment
    for word in words:
        comment.append(p_stemmer.stem(word))
    
    # return a string of stemmed words    
    return ' '.join(comment)


def p_stem_column(df,column):
    '''
    Applies PorterStemmer to all words in a column
    
    Arguments:
    --------------
    df : Dataframe
    
    column : Series
    '''
    df['stemmed comments'] = df[column].apply(p_stem)

    return df



# Looked at docstring of CountVectorizer to help with some of my docstring
def cvec_df(df,column,stop_words):
    '''
    Convert a collection of text documents from a DataFrame to a matrix of token 
    counts
    
    Arguments:
    --------------
    df : data
    
    column : Series
    
    stop_words : list, a list that is assumed to contain stop words, all of which
    will be removed from the resulting tokens.
    '''
    # instantiate CountVectorizer
    cvec = CountVectorizer(stop_words = stop_words)
    
    # define X
    X = df[column]
    
    # fit Count Vectorizer
    cvec.fit(X)
    
    # transform column data
    X = cvec.transform(X)
    
    # return dataframe of count vectorized comments
    return pd.DataFrame(X.todense(),columns = cvec.get_feature_names())   



# thanks for the stop word list charlie!
def stop_words(stop_word_list):
    '''
    Adds common stop words to you stop word list 

    Arguments:
    --------------
    stop_word_list : list, a stop word list that contains stop words relevant
    to the classification models
    
    '''
    # thanks for the stop word list charlie!
    charlie_stopword_list = ['anyone', 'take', 'beforehand', 'me', 'there', 
                         'will', 'from', 'except', 're', 'ever', 'several',
                         'twelve', 'move', 'in', 'until', 'they', 'bill', 
                         'each', 'which', 'con', 'ie', 'fire', 'two', 'how', 
                         'our', 'whole', 'thereupon', 'if', 'their', 'mostly', 
                         'thence', 'an', 'yourselves', 'made', 'neither', 
                         'thru','top', 'also', 'get', 'due', 'become','many', 
                         'whoever','somehow', 'few', 'much','whereafter', 
                         'whose', 'put','we', 'here','either','none', 'again', 
                         'becomes','had', 'keep', 'as', 'hers', 'forty', 'it', 
                         'not','thin','together', 'mine', 'without', 'both', 
                         'cant', 'most','third', 'others', 'under', 'at', 
                         'before', 'on', 'besides','whenever', 'because', 
                         'below', 'somewhere', 'has', 'or','between','perhaps', 
                         'whereas', 'de', 'eight', 'who','wherein', 'found', 
                         'afterwards', 'go', 'with', 'cannot','last', 'some', 
                         'seem', 'couldnt', 'and', 'other', 'what','since', 
                         'full', 'over', 'off', 'sometimes', 'beside', 'toward', 
                         'sometime', 'back', 'him', 'than', 'ltd', 'through', 
                         'first','whatever', 'his', 'might', 'was', 'elsewhere',
                         'do', 'every','among', 'yourself', 'being', 'bottom', 
                         'i','while','latterly','however','seemed','otherwise',
                         'four', 'detail', 'therein','name', 'thick', 'now', 
                         'see', 'inc', 'been', 'but', 'least','moreover', 'then', 
                         'whether','the','enough','three','herself','themselves', 
                         'my', 'would', 'herein','during','hereafter','wherever', 
                         'became','these','another','find','towards','something', 
                         'amount', 'front', 'still', 'were', 'whereupon','she', 
                         'above', 'to', 'less','is','etc','anything','ourselves', 
                         'eleven', 'all', 'everything', 'for', 'her', 'next', 
                         'could','any','five','by','interest','very','although', 
                         'indeed', 'ours', 'one', 'nowhere', 'already','always', 
                         'everyone', 'nine', 'upon', 'often', 'after','same', 
                         'throughout', 'when', 'cry', 'hereby','once','of','so', 
                         'side','fill','am','hence','twenty','call','becoming', 
                         'latter', 'well', 'may', 'part', 'nobody','into', 
                         'though', 'system', 'mill','hereupon','that','whereby', 
                         'have', 'almost', 'about', 'only', 'formerly','nor', 
                         'should', 'where', 'thereafter', 'yours', 'why','give', 
                         'anyway', 'therefore', 'seeming','hasnt','are','fifty', 
                         'yet', 'too', 'six','alone','himself','can','nothing', 
                         'thus', 'along','thereby','fifteen','beyond','further', 
                         'he', 'against', 'someone', 'us','un','whither','show', 
                         'rather', 'namely','empty','nevertheless', 'meanwhile', 
                         'even', 'former', 'its', 'out','please', 'via', 'your', 
                         'must','everywhere', 'a','onto', 'eg', 'hundred', 'up',
                         'noone', 'whom', 'seems', 'never', 'else','this','own', 
                         'describe','them','co','sixty','ten','serious','across', 
                         'you', 'myself','around', 'sincere', 'amongst', 'no', 
                         'anyhow','more','be','whence','such','anywhere','done', 
                         'itself','per','amoungst','down','behind','those',
                         'within']

    # add stopwords from nltk.corpus stop word list to inputted stopword list
    for word in stopwords.words('english'):
        stop_word_list.append(word)

    # add stopwords from charile's stop word list to inputted stopword list
    for word in charlie_stopword_list:
        stop_word_list.append(word)
        
    return stop_word_list



def ski_and_sb_stopword_list():
    # ski and snowboarding stop words list
    ski_and_sb_stopwords = ['ski','skiing','board','snowboard','snowboarding',
                            'boarding','skiers','snowboarders','snowboarder',
                            'skier','skis','boards','snowboards','www','https',
                            'com','splitboard','luge','heliskiing','downhill',
                            'heli-skiing','downhill','cross-country','surf',
                            'crosscountry','biathalon','snowcross','surfing',
                            'apres-ski','nordic combined','slalom','cross',
                            'parallel','alpine','speed skiing', 'telemark',
                            'ski jumping','mogul','newschool','ride','rode',
                            'riding','rides','ridden','skied','snowboarded',
                            'burton','rome','rider','deleted','poles','pole',
                            'sticks','stocks','snowplough','pizza','volkl']
    
    # return stop word list that combines stop_words function stop words and
    # ski_and_sb_stopwords
    return stop_words(ski_and_sb_stopwords)



def sentiment_analyzer(df,column):
    '''
    Use a collection of text documents from a DataFrame to add columns that provide
    summary statistics of sentiment analysis, as well as a binary classification of
    positive or negative sentiment.
    
    Arguments:
    --------------
    df : data
    
    column : Series
    '''
    # instantiate SentimentIntensityAnalyzer
    sentiment = SentimentIntensityAnalyzer()
    
    # loop to create a list of polarity scores for each row in the df
    df_sentiment_list = [sentiment.polarity_scores(text) for text in df[column]]
    
    # convert sentiment list into a df
    df_sentiment = pd.DataFrame(df_sentiment_list)
    
    
    sentiment_col = []

    # loop to see which comments are primarily negative or primarily positive
    for x in range(len(df_sentiment['neg'])):
        if df_sentiment['neg'][x] > df_sentiment['pos'][x] and df_sentiment['neg'][x] > .3:
            sentiment_col.append(-1)
        elif df_sentiment['pos'][x] > df_sentiment['neg'][x] and df_sentiment['pos'][x] > .3:
            sentiment_col.append(1)
        else:
            sentiment_col.append(0)
    
    # create a column labeling if the comment was primarily positive or negative
    df_sentiment['sentiment'] = sentiment_col
    
    # concatinate original dataframe with dataframe that analyzes the comments
    df = pd.concat([df,df_sentiment],axis = 1)
    
    return df



def pipeline_sentiment(df,column):
    '''
    Use a collection of text documents from a DataFrame to add columns that classifies 
    the columns as positive or negative sentiment, as well as the sentiment score.
    
    Arguments:
    --------------
    df : data
    
    column : Series
    '''
    # instantiate sentiment-analysis
    sent = pipeline('sentiment-analysis')
    
    # create a list of the sentiment analysis for each comment
    df_sent = [sent(df[column][x]) for x in range(len(df[column]))]
    
    # since sentiment analysis returns a dictionary within a list for each comment
    # we need to convert it into a list of all the dictionaries
    df_sents = [df_sent[x][0] for x in range(len(df_sent))]

    # convert the list into a Dataframe
    df_sentiments = pd.DataFrame(df_sents)
    
    # change column names
    df_sentiments.columns = ['pos/neg','score']
    
    # add sentiment analysis to the main Dataframe
    df = pd.concat([df,df_sentiments],axis = 1)
    
    df['pos/neg'] = df['pos/neg'].map({'POSITIVE': 1,'NEGATIVE': 0})

    return df



    # inspired by lesson 5.03 and adapted to work on series of comments

def freq_table(df,column,count=None):

    text = []
    
    # loop to turn all of the comments in a series into a single list
    # of words
    for comment in df[column]:
        for word in comment.split():
            text.append(word)
    
    freqs = FreqDist(text).most_common()
    prob = [round(x[1]/len(text),4) for x in freqs]

    freq = zip(freqs,prob)

    comp_freqs = []
    for s,p in freq:
        comp_freqs.append([s[0],s[1],p])

    comp_freqs.sort(key=lambda tup: tup[1], reverse=True) #Sort the list so it's in order, big to small

    if count == None:
        most = comp_freqs[:26]    
        hapax = comp_freqs[:-25:-1]
    else:
        most = comp_freqs[:count + 1]
        hapax = comp_freqs[:-count:-1]

    print('Most Common \t\t  Least Common')

    for i in zip(most, hapax):
        print(i[0], " "*(24-len(str(i[0]))),i[1])



def model_confusion_matrix(model,X,y):
    '''
    Use a model to create a confusion matrix and print out model results
    
    Arguments:
    --------------
    model : model

    X : X, data predictions to be made on

    y : y, data for predictions to be compared to
    ''' 
    preds = model.predict(X)

    tn,fp,fn,tp = confusion_matrix(y,preds).ravel()
    print(f'True Positives: {tp}')
    print(f'True Negatives: {tn}')
    print(f'False Positives: {fp}')
    print(f'False Negatives: {fn}')
    print('')
    print(f'Accuracy: {(tp+tn)/(tp+tn+fp+fn)}')
    print(f'Misclassification Rate: {1-(tp+tn)/(tp+tn+fp+fn)}')
    print(f'Precision: {tp/(tp+fp)}')
    print(f'Recall: {tp/(tp+fn)}')
    print(f'Specificity: {tn/(tn+fp)}')

    return plot_confusion_matrix(model.best_estimator_,X,y, 
                      display_labels= ['Skier', 'Snowboarder'], 
                      cmap = 'Blues_r');