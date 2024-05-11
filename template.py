import math

class NaiveBayesClassifier:

    def __init__(self, classes):

        self.classes = classes
        self.class_word_counts = {} 
        self.class_counts = {"positive":0,"negative":0,"neutral":0}
        self.count = 0
        self.likelihood = {}
        self.prior = {}
        self.vocab = set()

    def train(self, data):
        for features, label in data:
            self.count += 1

            self.class_counts[label] +=1

            for word in features :
                self.vocab.add(word)
                if word not in self.class_word_counts:
                    self.class_word_counts[word] = {"positive":0,"negative":0,"neutral":0}

                self.class_word_counts[word][label] += 1

        for word in self.vocab:
            self.likelihood[word] = {}
            for c in self.classes:
                self.likelihood[word][c] = self.calculate_likelihood(word,c)

        self.calculate_prior()
                
                
    def calculate_prior(self):
        for c in self.classes:
            self.prior[c] =  math.log(self.class_counts[c]/self.count)

    def calculate_likelihood(self, word, label):
        if word in self.vocab:
            return math.log((self.class_word_counts[word][label]+1) / (self.class_counts[label] + len(self.vocab)))
        else:
            return math.log( 1 / (self.class_counts[label] + len(self.vocab)))

    def classify(self, features):
        best_class = None 
        max = float('-inf')
        classValue = {}
        for c in self.classes:
            classValue[c] = 0
            for f in features:
                if f in self.vocab:
                    classValue[c] += self.likelihood[f][c]
                else:
                   classValue[c] += self.calculate_likelihood(f,c)
            classValue[c] += self.prior[c]

        for val in classValue:
            if classValue[val] > max :
                best_class = val
                max = classValue[val]
       

        return best_class
