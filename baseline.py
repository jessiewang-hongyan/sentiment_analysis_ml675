import nltk
from nltk.corpus import stopwords
from nltk.data import find
import re
import string
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import gensim
from preprocess import read_data, stats_by_category


'''Clean data.'''
nltk.download('stopwords')
nltk.download('punkt')
# nlp = spacy.load('en_core_web_sm')
stop_words = stopwords.words('english') + ['u', 'ur', '4', '2', 
             'im', 'dont', 'doin', 'ure']

def clean_text(text):
    """
    This function changes all characters to lowercase, removes text 
    in square brackets, links, punctuation and words containing 
    numbers. It removes stop-words and lemmatizes the text and   
    returns the cleaned text
    """
    # lowercase
    text = text.lower()
    # remove text in [] - emoji
    text = re.sub('\[.*?\]', '', text)
    # remove quotation
    text = re.sub('\'(@.*)*\'', '', text)
    # remove links
    text = re.sub('https?://\S+|www\.\S+', '', text)
    # remove text in <> - html tags
    text = re.sub('<.*?>+', '', text)
    # remove username
    text = re.sub('@(/w)*', '', text)
    # remove puncturation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # remove newline
    text = re.sub('\n', '', text)
    # remove mixture of letters and digits
    text = re.sub('\w*\d\w*', '', text)
    text = ''.join([c for c in text if c not in string.punctuation])
    tokens = re.split('\W+', text)
    # text = ' '.join([word for word in tokens if word not in    
    #        stop_words])
    text = ' '.join([word for word in tokens])

    # text = nlp(text)
    # text = ' '.join([word.lemma_ for word in text])

    # stemming
    stem = nltk.stem.SnowballStemmer('english')
    stemmed = []

    for token in nltk.word_tokenize(text):
        stemmed.append(stem.stem(token))

    return stemmed

class vocab_count():
    def __init__(self):
        self.vocab = dict()

    def set_vocab_count(self, sentences, labels):
        # word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
        # model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)

        for s, l in zip(sentences,labels):
            for word in s:
                if word not in self.vocab.keys():
                    self.vocab[word] = np.zeros((3,))
                self.vocab[word][l] += 1
        return self

    def get_vocab_count(self, sentence):
        vec = np.zeros((3,))
        for word in sentence:
            if word in self.vocab.keys():
                vec += self.vocab[word]
        return vec

    def get_count_matrix(self, X):
        counted_X = []
        for x_i in X:
            counted_X.append(self.get_vocab_count(x_i))
        matrix_X = np.stack(counted_X, axis=0)
        return matrix_X


if __name__ == '__main__':
    X, y = read_data()
    ratios = stats_by_category(X, y)
    weights = [1/r for r in ratios]
    normalizer = sum(weights)
    weights = [w/normalizer for w in weights]
    weights_dict = dict()
    for i, w in enumerate(weights):
        weights_dict[i] = w

    cleaned_X = []
    for tw in X:
        cleaned_X.append(clean_text(tw))
    
    X_train, X_test, y_train, y_test = train_test_split(cleaned_X, y, test_size=0.2)

    my_vocab = vocab_count()
    my_vocab.set_vocab_count(X_train, y_train)
    X_train_matrix = my_vocab.get_count_matrix(X_train)
    y_train_matrix = np.array(y_train)
    X_test_matrix = my_vocab.get_count_matrix(X_test)
    y_test_matrix = np.array(y_test)

    lr_model = LogisticRegression(class_weight=weights_dict).fit(X_train_matrix, y_train_matrix)
    pred = lr_model.predict(X_test_matrix)

    acc = accuracy_score(y_test_matrix, pred)
    conf = confusion_matrix(y_test_matrix, pred)

    print(f'Logistic Accuracy: {acc}')
    print(f'Logistic Confusion Matrix: \n{conf}')





    



    
    
    