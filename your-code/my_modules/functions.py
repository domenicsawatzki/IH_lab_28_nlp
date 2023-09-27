import re
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize


from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet

nltk.download('omw-1.4')
nltk.download('wordnet') # wordnet is the most well known lemmatizer for english
nltk.download('punkt')
nltk.download('stopwords')


def clean_up(s):
    """
    Cleans up special characters, numbers, and URLs from a string.

    Args:
        s: The string to be cleaned up.

    Returns:
        A string that has been cleaned up.
    """
    # Step 1: Remove URLs
    s = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', s)
    
    # Step 2: Replace numbers and special characters with a whitespace
    s = re.sub(r'[^a-zA-Z\s]', ' ', s)
    
    # Step 3: Replace multiple spaces with a single space
    s = re.sub(r'\s+', ' ', s)
    
    return s.lower().strip()





def tokenize(s):
    """
    Tokenize a string.

    Args:
        s: String to be tokenized.

    Returns:
        A list of words as the result of tokenization.
    """
    tokens = word_tokenize(s)
    return [word.lower() for word in tokens if word.isalnum()]



def get_wordnet_pos(word):
    tag_dict = {"J": wordnet.ADJ, 
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV}
    tag = nltk.pos_tag([word], lang='eng')[0][1][0].upper()
    return tag_dict.get(tag, wordnet.NOUN)

def stem_and_lemmatize(l):
    """
    Perform stemming and lemmatization on a list of words.

    Args:
        l: A list of strings.

    Returns:
        A list of strings after being stemmed and lemmatized.
    """
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in l]
    return lemmatized 


from nltk.corpus import stopwords

def remove_stopwords(l):
    """
    Remove English stopwords from a list of strings.

    Args:
        l: A list of strings.

    Returns:
        A list of strings after stop words are removed.
    """


    text_no_sw = [word for word in l if not word in stopwords.words('english')]
    
    return text_no_sw