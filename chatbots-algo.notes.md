Communication :
- use slack api or other apis
https://slack.dev/python-slackclient/
- use real time messaging:
https://api.slack.com/rtm


### bots
Apple’s Siri, Amazon’s Alexa, and Microsoft’s Cortana
- 2 types:
 - Rule-Based : answers -> pre-determined rules -> trained
 - Self-learning : AI, ML to learn. Natural language processing(https://www.nltk.org/)
   - 2 types:
    - retrieval
    - generative


### NLTK
https://www.nltk.org/
cite: Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.

```
pip install --user -U nltk
pip install --user -U numpy
```

- Installing NLTK Data -> datasets, models
https://www.nltk.org/data.html
 -  `python -m nltk.downloader popular`
 or
 - `import nltk; nltk.download('popular')` --> via interpreter

/home/s1m0x/nltk_data


#### examples:
- tokenize and tag text
sentence = """ thank you mohamed, you are awesome """
tokens = nltk.word_tokenize(sentence)
tagged = nltk.pos_tag(tokens)
tagged[0:6]

- identify named entities
entities = nltk.chunk.ne_chunk(tagged)

- display a parse tree
```
from nltk.corpus import treebank
t = treebank.parsed_sents('wsj_0001.mrg')[0]
t.draw()
```

https://www.nltk.org/api/nltk.html

----modules---
https://www.nltk.org/api/nltk.tokenize.html#module-nltk.tokenize

- tokenizer -> divide string into substrings

https://towardsdatascience.com/build-your-first-chatbot-using-python-nltk-5d07b027e727
=> hardcoded probable questions and answers
Goals: ?
1. Understand who is the target audience
2. Understand the Natural Language of the communication.
3. Understand the intent or desire of the user
4. provide responses that can answer the user

- use https://www.nltk.org/api/nltk.chat.html
- Reflections -> dict has input and corresponding output.
ex:
```
reflections = {
  "i am"       : "you are",
  "i was"      : "you were",
  "i"          : "you",
  "i'm"        : "you are",
  "i'd"        : "you would",
  "i've"       : "you have",
  "i'll"       : "you will",
  "my"         : "your"
}
```
or
```
my_dummy_reflections= {
    "go"     : "gone",
    "hello"    : "hey there"
}
```
- use your reflections
chat = Chat(pairs, my_dummy_reflections)

ex: see chat.py -> /home/s1m0x/Desktop/pfe/Devs/botchat/ntlk/

- nltk.chat chatbots -> regex keywords in questions

#### Chatbot with deep learning

https://towardsdatascience.com/how-to-create-a-chatbot-with-python-deep-learning-in-less-than-an-hour-56a063bdfc44
https://github.com/jerrytigerxu/Simple-Python-Chatbot

- Agenda
Libraries & Data
Initializing Chatbot Training
Building the Deep Learning Model
Building Chatbot GUI
Running Chatbot
Conclusion
Areas of Improvement

- files:
train_chatbot.py -> Keras sequential neural network to create a model
chatgui.py -> cleanning respons based on prediction & offers gui
classes.pkl -> diff types of classes of responses
words.pkl -> diff words for pattern recognition
intents.json -> js objects, list diff tags corresp diff types of word patterns
chatbot_model.h5 -> actual model trained by train_chatbot.py
keras -> deep learning framework

aside notes: Se technical words:
---

https://www.datacamp.com/community/tutorials/stemming-lemmatization-python

- Stop words
 - do not contain important significance, mostly commonly words, ex as, be, are..
 - are filtered from search queries
- Stemming:
 - is the process of reducing inflection in words to their root forms
 - mapping a group of words to the same stem(root)

- Lemmatization :
 - reduces the inflected words properly ensuring that the root word belongs to the language
 - root word -> called Lemma -> canonical form of word
  - ex: talks, talking, talked -> lemma : talk
 - nltk -> WordNet Lemmatizer (WordNet db)

```
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
wordnet_lemmatizer.lemmatize(word, pos="v") # POS (part-of-speech) tp get root of word
```

- tokenizer -> separe sentences into words
word_tokenize(sentence) --> (`from nltk.tokenize import sent_tokenize, word_tokenize`)
- tokenize the line -> then stem the word

 - nltk english stemmers -> PorterStemmer and LancasterStemmer
 -  SnowballStemmers as a language to create to create non-English stemmers
 - ISRIStemmer is an Arabic stemmer

```
from nltk.stem.snowball import SnowballStemmer
englishStemmer=SnowballStemmer("english")
englishStemmer.stem("having")
```

- building the deep learning model:
 - Sequential model -> in keras -> simplest neural network, multilayer perceptron
https://keras.io/guides/sequential_model/
 - train model with stochastic gradient descent
 - deep learningn frmwrks -->
  - tensorflow, Apache Spark, PyTorch, Sonnet
