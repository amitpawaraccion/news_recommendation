
import numpy as np
np.random.seed(2018)
import pandas as pd
import matplotlib.pyplot as plt #Visualization 
import seaborn as sns # High Level Visualization and Statistical Graphs
sns.set_theme(style="ticks", color_codes=True)

from scipy import stats # Scintific Python for Statistics
import spacy # Rule Based Matching, Pipelines, Embedding, Viduslization of Text
import pickle # To store and Use Models
import string # For String Operations
import re, time # Regular Expression and Time
import unicodedata # For Different Data Formats
from bs4 import BeautifulSoup # For Web Scrapping

import nltk # Natural Language Tool Kit
from nltk.corpus import stopwords # To filter out most common words
from nltk.tokenize import RegexpTokenizer #  Splitting into Smaller Units
from nltk.stem.porter import PorterStemmer, StemmerI #  Information Retrival and Extraction
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer, SnowballStemmer # Large Lexical Database and Stemming Algorithms
nltk.download('stopwords')
nltk.download('wordnet')

import texthero as hero  #Texthero makes it easy to apply TF-IDF to the text in the dataframe
from texthero import preprocessing # Efficient Pre-processing

import gensim # Unsupervised Topic Modeling and NLP
from gensim.test.utils import common_texts # Package that notifies Common Texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument# Doc2Vec algorithm gives a document another floating word-like vector , called a doc-vector 
from gensim.utils import simple_preprocess # Gives Unicode STrings lowercased 
from gensim.parsing.preprocessing import STOPWORDS # To filter out most common words
from gensim import corpora

from sklearn.decomposition import LatentDirichletAllocation #Similarity in unobserved Groups
from sklearn.datasets import make_multilabel_classification # Multi Output Scenarios
from sklearn.feature_extraction.text import TfidfVectorizer
from gsdmm import MovieGroupProcess
from tqdm import tqdm

df = pd.read_csv("output.csv", error_bad_lines=False)
df_s = df
df.columns = [x.upper() for x in df_s.columns]
df.count()
df.info(memory_usage='deep')
df.describe()
df.info()

data= df
data_text = data[['ACTUAL']]
data_text['index'] = data_text.index
documents = data_text

# convert string of tokens into tokens list
df['split'] = df.ACTUAL.apply(lambda x:  re.split('\s', x))
df.head()
# create list of  token lists
docs = df['split'].tolist()
docs[:3]


# Train STTM model
#    K = number of potential topics
#    alpha = controls completeness
#    beta =  controls homogeneity 
#    n_iters = number of iterations
mgp = MovieGroupProcess(K=70, alpha=0.1, beta=0.5, n_iters=500)
vocab = set(x for doc in docs for x in doc)
n_terms = len(vocab)
y = mgp.fit(docs, n_terms)

# Save model
with open('media_model.model', 'wb') as f:
    pickle.dump(mgp, f)
    f.close()

filehandler = open('media_model.model', 'rb')
mgp = pickle.load(filehandler)

def top_words(cluster_word_distribution, top_cluster, values):
    '''prints the top words in each cluster'''
    for cluster in top_cluster:
        sort_dicts =sorted(mgp.cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:values]
        print('Cluster %s : %s'%(cluster,sort_dicts))
        print(' — — — — — — — — —')
        
def cluster_importance(mgp):
    '''returns a word-topic matrix[phi] where each value represents
    the word importance for that particular cluster;
    phi[i][w] would be the importance of word w in topic i.
    '''
    n_z_w = mgp.cluster_word_distribution
    beta, V, K = mgp.beta, mgp.vocab_size, mgp.K
    phi = [{} for i in range(K)]
    for z in range(K):
        for w in n_z_w[z]:
            phi[z][w] = (n_z_w[z][w]+beta)/(sum(n_z_w[z].values())+V*beta)
    return phi

def topic_allocation(df, docs, mgp, topic_dict):
    '''allocates all topics to each document in original dataframe,
    adding two columns for cluster number and cluster description'''
    topic_allocations = []
    for doc in tqdm(docs):
        topic_label, score = mgp.choose_best_label(doc)
        topic_allocations.append(topic_label)

    df['cluster'] = topic_allocations
    df['topic_name'] = df.cluster.apply(lambda x: get_topic_name(x, topic_dict))
    print('Complete. Number of documents with topic allocated: {}'.format(len(df)))

def get_topic_name(doc, topic_dict):
    '''returns the topic name string value from a dictionary of topics'''
    topic_desc = topic_dict[doc]
    return topic_desc

doc_count = np.array(mgp.cluster_doc_count)
print('Number of documents per topic :', doc_count)
print('*'*20)

# topics sorted by the number of documents they are allocated to
top_index = doc_count.argsort()[-10:][::-1]
print('Most important clusters (by number of docs inside):', top_index)
print('*'*20)

phi = cluster_importance(mgp) # initialize phi matrix
# 'coronavirus' term importance for cluster 1 and 0 
print(phi[1]['KINGS'])
print(phi[0]['PLANET'])

topic_indices = np.arange(start=0, stop=len(doc_count), step=1)
top_words(mgp.cluster_word_distribution, topic_indices, 5)

# define dictionary topics in same sequential order
# as resulting clusters from gsdmm model 
topic_dict = {}
topic_names = ['Topicone',
               'Topictwo',
               'Topicthree',
               'Topicfour',
               'Topicfive',
               'Topicsix',
               'Topicseven',
               'Topiceight',
               'Topicnine',
               'Topicten']

for i, topic_num in enumerate(topic_indices):
    topic_dict[topic_num]=topic_names[i]
    
    
# allocate topics to original data frame 
topic_allocation(df, docs, mgp, topic_dict)
df

# helper functions to extract all data needed to create bubble charts for exploring words in each topic
def top_words_dict(cluster_word_distribution, top_cluster, n_words):
    '''returns a dictionary of the top n words and the number of docs they are in;
    cluster numbers are the keys and a tuple of (word, word count) are the values'''
    top_words_dict = {}
    for cluster in top_cluster:
        top_words_list = []
        for val in range(0, n_words):
            top_n_word = sorted(mgp.cluster_word_distribution[cluster].items(), 
                                key=lambda item: item[1], reverse=True)[:n_words][val]    #[0]
            top_words_list.append(top_n_word)
        top_words_dict[cluster] = top_words_list

    return top_words_dict

def get_word_counts_dict(top_words_nclusters):
    '''returns a dictionary that counts the number of times a word 
    appears only in the top n words list across all the clusters;
    words are the keys and a count of the word is the value'''
    word_count_dict = {}
    for key in top_words_nclusters:
        words_score_list = []
        for word in top_words_nclusters[key]:
            if word[0] in word_count_dict.keys():
                word_count_dict[word[0]] += 1
            else:
                word_count_dict[word[0]] = 1
    return word_count_dict

def get_cluster_importance_dict(top_words_nclusters, phi):
    '''returns a dictionary that of all top words and their cluster
    importance value for each cluster;
    cluster numbers are the keys and a list of word 
    importance computed scores are the values'''
    cluster_importance_dict = {}
    for key in top_words_nclusters:
        words_score_list = []
        for word in top_words_nclusters[key]:
            importance_score = phi[key][word[0]]
            words_score_list.append(importance_score)
        cluster_importance_dict[key] = words_score_list
    return cluster_importance_dict

def get_doc_counts_dict(top_words_nclusters):
    '''returns a dictionary of only the doc counts of each top n word for each cluster;
    cluster numbers are the keys and a list of doc counts are the values'''
    doc_counts_dict = {}
    for key in top_words_nclusters:
        doc_counts_list = []
        for word in top_words_nclusters[key]:
            num_docs = word[1]
            doc_counts_list.append(num_docs)
        doc_counts_dict[key] = doc_counts_list
    return doc_counts_dict

def get_word_frequency_dict(top_words_nclusters, word_counts):
    '''returns a dictionary of only the number of occurences across all 
    clusters for each word in a particular cluster's top n words;
    cluster numbers are the keys and a list of 
    word occurences counts are the values'''
    word_frequency_dict = {}
    for key in top_words_nclusters:
        words_count_list = []
        for word in top_words_nclusters[key]:
            words_count_list.append(word_counts[word[0]])
        word_frequency_dict[key] = words_count_list

    return word_frequency_dict

# declare any static variables needed 
nwords = 10
nclusters = len(topic_names)
phi = cluster_importance(mgp)

# define and generate dictionaries that hold each topic number and its values
top_words = top_words_dict(mgp.cluster_word_distribution, topic_indices, nwords)
word_count = get_word_counts_dict(top_words)
word_frequency = get_word_frequency_dict(top_words, word_count)
cluster_importance_dict = get_cluster_importance_dict(top_words, phi)
    
# add all values for each topic to a list of lists
rows_list = []
for cluster in range(0, nclusters):
    topic_name = topic_names[cluster]
    words = [x[0] for x in top_words[cluster]]
    doc_counts = [x[1] for x in top_words[cluster]]
    
    # create a list of values which represents a 'row' in our data frame 
    rows_list.append([int(cluster), topic_name, words, doc_counts, 
                     word_frequency[cluster], cluster_importance_dict[cluster]])
        
topic_words_df = pd.DataFrame(data=rows_list, 
                              columns=['cluster','topic_name', 'top_words',
                                        'doc_count', 'num_topic_occurrence', 'word_importance'])

#  STMM modeltrain
mgp = MovieGroupProcess(K=10, alpha=0.1, beta=0.1, n_iters=30)
vocab = set(x for doc in docs for x in doc)
n_terms = len(vocab)
y = mgp.fit(docs, n_terms)
# save model
with open('dumps/trained_models/10clusters.model', 'wb') as f:
    pickle.dump(mgp, f)
    f.close()

sns.catplot(x="ACTUAL", data=df)
df['ACTUAL'].apply(len).plot(kind='line', color='green', figsize=(16,8))
# (kind = 'hist')
df.groupby('ACTUAL')['ACTUAL'].count().plot(kind='line', color='green', figsize=(16,8))
plt.xlabel('NAmes')
plt.ylabel('Count')
plt.show()
