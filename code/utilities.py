from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()


from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score,\
            f1_score, classification_report, roc_curve,auc, roc_auc_score


import nltk

nltk.download('words', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)

class nlp_pipe:
    
    def __init__(self, vectorizer=CountVectorizer(), cleaning_function=None,
                 tokenizer=None, stemmer=None):

        self.stemmer = stemmer
        self.tokenizer = tokenizer
        self.cleaning_function = cleaning_function
        self.vectorizer = vectorizer
        self.lemmatizer = WordNetLemmatizer()
        self._is_fit = False    
    def fit_transform(self, text_to_fit_on):
        """
        Cleans the data and then fits the vectorizer with
        the user provided text
        """

        text_to_fit_on = text_to_fit_on.apply(self.normailze_text).apply(self.tokenize_text).apply(self.remove_stopwords).apply(self.stem_lem_words).apply(self.listToString)

        self.vectorizer.fit(text_to_fit_on)
        return self.vectorizer.transform(text_to_fit_on)
            
    def tokenize_text(self,text):
    
        #Tokenize words
        tokens = nltk.word_tokenize(text)

        return tokens
    def normailze_text(self, text):
    
        # Convert to lowercase
        text = text.lower()

        # Remove extra characters
        text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))

        # Remove punctuation characters
        text = re.sub(r"[^a-zA-Z0-9]", " ", text) 

        # Remove symbols
        text = re.sub(r'[^A-Za-z\s]',r'',text)
        text = re.sub(r'\n',r'',text)
        # Remove two characters
        resulst =[]
        for i in text.split(' '):
            resulst.append(re.sub(r'^\w{0,2}$',r'',i))
        text = ' '.join(resulst) 
        return text
    def remove_stopwords(self, tokens):
    
        stop_words = stopwords.words('english')
        token_list = []

        for word in tokens:
            if not word in stop_words:
                token_list.append(word)


        return token_list
    
    def stem_lem_words(self, tokens):

        #Lemmatizing tokens
        tokens = [self.lemmatizer.lemmatize(token, pos='v') for token in tokens]

        return tokens
    
    def listToString(self, s): 
    
        # initialize an empty string
        str1 = " " 
        # return string  
        return (str1.join(s))

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from rich.console import Console
from rich.table import Table

class ml:
    dic_mach = {'KNN_5':KNeighborsClassifier(),
        'Log_Reg': LogisticRegression(), 
        'DTC':DecisionTreeClassifier(), 
        'RFC': RandomForestClassifier()}
    console = Console()

    def __init__(self, machines=dic_mach):
        self.machines = machines
        self.__is_fit = False
        self.__is_predict = False
        self.__predicted_values = None
        self.__y_pred = None
        self.__y_test = None
    def fit(self, x_train, y_train):
        dictionary = dict()
        for key, values in self.machines.items():
            values.fit(x_train, y_train)
        self.__is_fit = True
    
    def predict(self, x_test):
        if  not self.__is_fit:
            raise TypeError("You have to run \"fit()\" funciton first")
        dictionary = dict()
        for key, values in self.machines.items():
            dictionary[key] = values.predict(x_test)
        self.__is_predict = True
        self.__y_pred = dictionary
        self.__x_test = x_test

        return dictionary

    def plot_heatmap(self, y_test, rows= 2, columns=2,figsize=(13,13)):
        if  not self.__is_predict:
            raise TypeError("You have to run \"predict()\" funciton first")

        fig = plt.figure(figsize=figsize)
        fig.suptitle('Heat Map for Confusion Matrixes')
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        for index, (key, value) in enumerate(self.__y_pred.items()):
            ax = fig.add_subplot(rows, columns, index+1)
            sns.heatmap(confusion_matrix(y_test,self.__y_pred[key]),ax=ax, annot=True, fmt=".0f",cbar=False)
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels'); 
            ax.set_title(key)
            ax.xaxis.set_ticklabels(['False', 'True'])
            ax.yaxis.set_ticklabels(['False', 'True'])
        self.__y_test = y_test

    def get_info(self):
        if  not self.__is_predict:
            raise TypeError("You have to run \"predict()\" funciton first")

        table = Table(title="Models Results", show_header=True, header_style="bold magenta")
        table.add_column("Model", style="dim", width=20)
        table.add_column("Accuracy", justify='center')
        table.add_column("Precision", justify='center')
        table.add_column("Recall", justify='center')
        table.add_column("F1", justify='center')
        for index, (key, value) in enumerate(self.__y_pred.items()):
            table.add_row(
                key, 
                str(round(accuracy_score(self.__y_test, value)* 100,2)), 
                str(round(precision_score(self.__y_test, value)* 100,2)),
                str(round(recall_score(self.__y_test, value)* 100,2)),
                str(round(f1_score(self.__y_test, value)* 100,2)),
            )
        self.console.print(table)

    def plot_roc(self):
        if  not self.__is_predict:
            raise TypeError("You have to run \"predict()\" funciton first")
        plt.figure(figsize=(20, 10))
        for index, (key, value) in enumerate(self.machines.items()):
            y_pred_proba = value.predict_proba(self.__x_test)[::,1]
            fpr, tpr, _ = roc_curve(self.__y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{key} = {roc_auc_score(self.__y_test, self.__y_pred[key]):.2f}')



        xx = np.linspace(0,1, 100000)
        plt.plot(xx, xx, linestyle='--')

        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')

