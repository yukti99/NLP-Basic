# -*- coding: utf-8 -*-
"""
@author: yukti
"""


import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

speech = """Respected teachers and my dear friends. Today, I, Yukti of Class XII of Amity International School, Noida, am here to share my views on the topic, Importance of Cleanliness. It has been rightly said that - Cleanliness is next to Godliness.The term cleanliness implies the absence of dust, dirt garbage or waste, etc. If one wants to be healthy and fit, a healthy environment is necessary. It is important for everyone’s mind and body as well as spirits like water and oxygen.The dirty environment affects the health of the people. The state of cleanliness reflects the character of its citizens. We can take a live example of today when people travel in a metro train, they never litter their surroundings, but once they come out of that vicinity, they hardly care. This habit shows their character. If we keep our surroundings clean, it will help in the nation-building exercise too as it will attract more tourists in the country.Awareness of the cleanliness is the need of the hour in our country where diseases like viral fever, swine flu, malaria, jaundice, etc. are spreading fastly. The awareness camps should be organised by the government, private organisations, and NGOs, and people to make the areas neat and clean. People in society and the community should organise rallies."""


sentences = nltk.sent_tokenize(speech)
# creating object for doing lemmatization
# will create meaningful words unlike stemming
lemmatizer = WordNetLemmatizer()

for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [lemmatizer.lemmatize(w) for w in words if w not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)
