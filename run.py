from template import NaiveBayesClassifier
import csv
import re
import nltk
import time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

def preprocess(tweet_string):

    tweet_string = re.sub(r'http+|www+|https+', '', tweet_string)
    tweet_string = re.sub(r'@+', '', tweet_string)
    tweet_string = re.sub(r'(.)\1+', r'\1\1', tweet_string)                  
    tweet_string = re.sub(r"\s+", " ", tweet_string)                           
    tweet_string = tweet_string.lower()

    tokens = word_tokenize(tweet_string)

    words = []
    stop_words = set(stopwords.words('english'))
    special_chars = set('!@#$%^&*()-_=+[{]};:\'",<.>/?\\|`~0123456789')
    for word in tokens :
        if word not in stop_words :
            valid = True
            for char in special_chars:
                if char == word:
                    valid = False
                    break
            if valid :
                words.append(word)

    return words

def load_data(data_path):
    data = []
    
    with open(data_path, 'r') as file:
        csv_reader = csv.reader(file)
        
        next(csv_reader, None)
        
        for row in csv_reader:
            words = preprocess(row[2])
            data.append((words,row[4]))
    
    return data

def eval():
    with open('eval_data.csv', 'r') as file:
        csv_reader = csv.reader(file)
        
        next(csv_reader, None)
        valid = 0
        total = 0
        for row in csv_reader:
            c = nb_classifier.classify(preprocess(row[2]))
            if c == row[4]:
                valid +=1
            total+=1
        print("eval compeleted")
        print(f"Accuracy of the model: {(valid/total) * 100}") 

def test():
    result = open("resuⅼt.txt","w+")
    with open('test_data_nolabel.csv', 'r') as file:
        csv_reader = csv.reader(file)
        
        next(csv_reader, None)
        for row in csv_reader:
            c = nb_classifier.classify(preprocess(row[2]))
            result.write(f"{c}\n")
    result.close()

train_data_path = 'train_data.csv'
classes = ['positive', 'negative', 'neutral']
nb_classifier = NaiveBayesClassifier(classes)
start_time = time.time()
nb_classifier.train(load_data(train_data_path))
end_time = time.time()
elapsed_time = end_time - start_time
print("train compeleted")
print(f"train time :{elapsed_time}")
eval()
test()
print("test compeleted")
