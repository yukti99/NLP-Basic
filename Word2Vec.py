# -*- coding: utf-8 -*-
"""
@author: yukti
"""


import nltk
import re
from gensim.models import Word2Vec
from nltk.corpus import stopwords

speech = """Respected teachers and my dear friends. Today, I, Yukti of Class XII of Amity International School, Noida, am here to share my views on the topic, Importance of Cleanliness. It has been rightly said that - Cleanliness is next to Godliness.The term cleanliness implies the absence of dust, dirt garbage or waste, etc. If one wants to be healthy and fit, a healthy environment is necessary. It is important for everyone’s mind and body as well as spirits like water and oxygen.The dirty environment affects the health of the people. The state of cleanliness reflects the character of its citizens. We can take a live example of today when people travel in a metro train, they never litter their surroundings, but once they come out of that vicinity, they hardly care. This habit shows their character. If we keep our surroundings clean, it will help in the nation-building exercise too as it will attract more tourists in the country.Awareness of the cleanliness is the need of the hour in our country where diseases like viral fever, swine flu, malaria, jaundice, etc. are spreading fastly. The awareness camps should be organised by the government, private organisations, and NGOs, and people to make the areas neat and clean. People in society and the community should organise rallies."""

# DATA PREPROCESSING AND CLEANING

s = re.sub(r'\[[0-9]*\]',' ',speech)
# removing unnecessary spaces: \s to find  whitespace character - space or tab
s = re.sub(r'\s+',' ',s)
# converting to lower case
s = s.lower()
# removing digits(single digits)
s = re.sub(r'\d',' ',s)

# TOKENISATION & REMOVING STOP WORDS
s = nltk.sent_tokenize(s)
sentences = [nltk.word_tokenize(sen) for sen in s]

for i in range(0,len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]

# WORD2VEC MODEL
    
# second paraments means if word iis present less than <min_count> times, skip the word
model = Word2Vec(sentences,min_count=1)

words = model.wv.vocab
# to find word vectors (with by default 100 dimensions)
vector = model.wv['clean']

# To find the most similar words
similar = model.wv.most_similar('dust')




