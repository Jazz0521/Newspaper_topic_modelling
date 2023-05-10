import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition
#import matplotlib.pyplot as plt
import os
import logging
import numpy as np
import re
import nltk
nltk.download('wordnet')
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import gensim
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer  
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
stop_words=[nltk.corpus.stopwords.words('english')]

def clean_text(headline):
      le=WordNetLemmatizer()
      word_tokens=word_tokenize(headline)
      tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
      cleaned_text=" ".join(tokens)
      return cleaned_text

def main():
    curr_dir = os.getcwd()
    logging.basicConfig(filename= curr_dir + '/log_files/tm1.log',level=logging.INFO,format='%(asctime)s %(lineno)s: %(message)s')
    logging.info('Start.')
    df=pd.read_csv(r'archive/Articles.csv', encoding="utf-8")
    #df=pd.read_csv(r'new.csv')
    logging.info('Read csv')
    # Remove punctuation
    df['text_processed'] = \
    df['Article'].map(lambda x: re.sub('[^a-zA-Z]', ' ', x))
    logging.info('removed special characters and digits from text')
    # Convert to lowercase
    df['text_processed'] = \
    df['Article'].map(lambda x: x.lower())
    logging.info('converted text to lowercase')
    # Print out the first rows of papers
    logging.info(df['text_processed'].head())
    #logging.info(df.head())
    logging.info(df.shape)

    papers=df.copy()
    
    df['cleaned_text']= df['text_processed'].apply(clean_text)
    logging.info('lematization and tokenization doone')
    #stop_words_set = frozenset(stop_words)
    # Get the existing English stopwords from the TfidfVectorizer
    existing_stopwords = set(TfidfVectorizer(stop_words='english').get_stop_words())

    # Define your custom list of stopwords to add
    custom_stopwords = ["said", "claimed","openc"]

    # Concatenate the existing stopwords with the custom stopwords
    extended_stopwords = list(existing_stopwords) + custom_stopwords

    vect =TfidfVectorizer(stop_words=extended_stopwords,max_features=1000)
    #stop_words = tuple(stop_words)
    vect_text=vect.fit_transform(df['cleaned_text'])
    logging.info("applied TfIdf vectorizer")
    from sklearn.decomposition import LatentDirichletAllocation
    lda_model=LatentDirichletAllocation(n_components=10,
    learning_method='online',random_state=42,max_iter=1) 
    lda_top=lda_model.fit_transform(vect_text)
    logging.info("applied lda ")
    print("Document 0: ")
    logging.info(" Document 0:")
    for i,topic in enumerate(lda_top[0]):
        print("----------------------------------------------")
        logging.info("----------------------------------------------------")
        print("Topic ",i,": ",topic*100,"%")
        logging.info("topic %d %f %%", i, topic*100)
    vocab = vect.get_feature_names_out()
    for i, comp in enumerate(lda_model.components_):
        vocab_comp = zip(vocab, comp)
        sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
        logging.info("----------------------------------------------------")
        print("----------------------------------------------")
        print("Topic "+str(i)+": ")
       
        
        logging.info("topics %s",i)
        for t in sorted_words:
            print(t[0],end=" ")
            logging.info(t[0])
            print(",")    
    
if __name__ == "__main__":
    main()
