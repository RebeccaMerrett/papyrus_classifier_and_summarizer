# papyrus_classifier_and_summarizer

To import the below: \n
pip install nltk
pip install scikit-learn
pip install numpy
pip install scipy
pip install networkx

import re
import nltk
nltk.download('punkt') #DOWNLOAD THIS ONCE WHEN FIRST RUNNING THE SCRIPT, THEN COMMENT OUT
nltk.download('averaged_perceptron_tagger') #DOWNLOAD THIS ONCE WHEN FIRST RUNNING THE SCRIPT, THEN COMMENT OUT
nltk.download('maxent_ne_chunker') #DOWNLOAD THIS ONCE WHEN FIRST RUNNING THE SCRIPT, THEN COMMENT OUT
nltk.download('words') #DOWNLOAD THIS ONCE WHEN FIRST RUNNING THE SCRIPT, THEN COMMENT OUT
from nltk import word_tokenize, pos_tag, ne_chunk
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import networkx as nx
