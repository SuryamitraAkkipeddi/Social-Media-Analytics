##############################################
####   Scraping APIs using Praw module  ######
##############################################

import praw
import datetime
import pymysql
import pymysql.cursors

reddit = praw.Reddit(user_agent='cis591_team7)',
                     client_id='5id0C-qmQ0pfBw', client_secret="6A26thWQVoQk5ecLbKhTST0mNMA",
                     username='cis591_team7', password='qwertyQWERTY123!@#')


subreddit = "pelotoncycle" #change this per collector

count = 1
for comment in reddit.subreddit(subreddit).stream.comments():
	commentID = str(comment.id).encode('utf8')
	author = str(comment.author).encode('utf8')
	timestamp = str(datetime.datetime.fromtimestamp(comment.created)).encode('utf8')
	replyTo = ""
	if not comment.is_root:
		replyTo = str(comment.parent().id).encode('utf8')
	else:
		replyTo = "-"
	threadID = str(comment.submission.id).encode('utf8')
	threadTitle = str(comment.submission.title).encode('utf8')
	msgBody = str(comment.body).encode('utf8')
	permalink = str(comment.permalink).encode('utf8')

	print("-------------------------------------------------------")
	print("Comment ID: " + str(comment.id))
	print("Comment Author: "+ str(comment.author))
	print("Timestamp: "+str(datetime.datetime.fromtimestamp(comment.created)))
	if not comment.is_root:
		print("Comment is a reply to: " + str(comment.parent().id))
	else:
		print("Comment is a reply to: -")
	print("Comment Thread ID: " + str(comment.submission.id))
	print("Comment Thread Title: " + str(comment.submission.title))
	print("Comment Body: " + str(comment.body))
	print("Comment Permalink: " + str(comment.permalink))

	db = pymysql.connect(host="localhost", user="root", passwd="qwertyQWERTY123!@#", db="peloton", charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
	cur = db.cursor()
	sqlStatement = "INSERT INTO " + subreddit + " (MsgID, Timestamp, Author, ThreadID, ThreadTitle, MsgBody, ReplyTo, Permalink) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
	inputData = (commentID, timestamp, author, threadID, threadTitle, msgBody, replyTo, permalink)
	cur.execute(sqlStatement, inputData )
	db.commit()
	db.close()

	print("Total messages collected from /r/"+subreddit+": " + str(count))
	count += 1

##############################################
####   Data Analysis and preprocessing  ######
##############################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Data Preprocessing and Feature Engineering
from textblob import TextBlob
import re
import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#Model Selection and Validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score


#Import data
train_messages = pd.read_csv('E://MSIM COURSES//591- Information Enabled Business Modeling//train_messages.csv', encoding='ISO-8859-1')
test_messages = pd.read_csv('E://MSIM COURSES//591- Information Enabled Business Modeling//test_peloton.csv', encoding='ISO-8859-1')


#Parse csv's to only extract column called 'message'
train_messages = train_messages[['label','message']]
test = test_messages['message']

#setup some variables to train your machine learning model
X = train_messages['message']
y = train_messages['label']
test = test_messages['message']

#Split your training set and test against it to generate some performance metrics (precision, recall) since it is labeled data
from sklearn.model_selection import train_test_split
msg_train, msg_test, label_train, label_test = train_test_split(train_messages['message'], train_messages['label'], test_size=0.2)

#Text pre-processing functions
def text_processing(message):

    #Generating the list of words in the message (hastags and other punctuations removed)
    def form_sentence(message):
        message_blob = TextBlob(message)
        return ' '.join(message_blob.words)
    new_message = form_sentence(message)

    #Removing stopwords and words with unusual symbols
    def no_user_alpha(message):
        message_list = [ele for ele in message.split() if ele != 'user']
        clean_tokens = [t for t in message_list if re.match(r'[^\W\d]*$', t)]
        clean_s = ' '.join(clean_tokens)
        clean_mess = [word for word in clean_s.split() if word.lower() not in stopwords.words('english')]
        return clean_mess
    no_punc_message = no_user_alpha(new_message)

    #Normalizing the words in messages
    def normalization(message_list):
        lem = WordNetLemmatizer()
        normalized_message = []
        for word in message_list:
            normalized_text = lem.lemmatize(word,'v')
            normalized_message.append(normalized_text)
        return normalized_message


    return normalization(no_punc_message)

#import nltk
#nltk.download()

#Machine Learning Pipeline
pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=text_processing)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
pipeline.fit(msg_train,label_train)

#This next line predicts against your split training set
predictions = pipeline.predict(msg_test)

#Print results of performance metrics (training set split experiment)
print(classification_report(predictions,label_test))
print ('\n')
print(confusion_matrix(predictions,label_test))
print(accuracy_score(predictions,label_test))

##############################################
####   Pedictions against Training set  ######
##############################################

#This next line predicts against your test set
predictions = pipeline.predict(test)

#Print out results of the predictions on unlabeled test set
print('\n\n\n') #Prints blank lines
print('')
i = 0
for msg in test:
	print("\"%s\" \n Was predicted as: %s\n\n" % (msg, predictions[i]))
	i = i + 1
	#can add some logic here to bin separately into postitive/negative/neutral

#########################################################
####   Word Cloud to get the high frequency terms  ######
#########################################################

# importing all necessery modules
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd

# Reads CSV file
df = pd.read_csv("E://MSIM COURSES//591- Information Enabled Business Modeling//test messages.csv")
comment_words = ' '
stopwords = set(STOPWORDS)
df.columns = df.columns.str.strip()
#counts = df['description'].value_counts()
#counts.index = counts.index.astype(str)
# iterate through the csv file

wordcloud = WordCloud(width = 800, height = 800,
				background_color ='white',
				stopwords = stopwords,
				min_font_size = 10).generate(comment_words)

# plot the WordCloud image
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)

#plt.show()




