#!/usr/bin/env python
# coding: utf-8

# # Text Preprocessing

# # 1. Lowercasing

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("IMDB Dataset.csv")


# In[3]:


df.head(5)


# In[4]:


df['review']=df['review'].str.lower()


# In[5]:


df.head(2)


# # 2. Remove HTML Tags

# In[6]:


import re

def remove_html_tags(text):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'',text)


# In[7]:


df['review']=df['review'].apply(remove_html_tags)


# In[8]:


df.head(2)


# # 3. Removing URLs

# In[9]:


text = "Check out my new notebook on https://www.colab.com/csv/notebook161745"


# In[10]:


def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)


# In[11]:


remove_url(text)


# # 4. Remove Punctuation

# In[12]:


import time, string
string.punctuation


# In[13]:


exclude = string.punctuation


# In[14]:


def remove_punc(text):
    for char in exclude:
        text = text.replace(char,'')
    return text


# In[15]:


text1 = 'string. with. Punctuation!?'


# In[16]:


remove_punc(text)


# # Another Method to remove punctuation with fast speed

# In[17]:


def remove_punc(text):
    return text.translate(str.maketrans('','',exclude))


# In[18]:


remove_punc(text1)


# In[19]:


df['review']=df['review'].apply(remove_punc)


# # 5. Chat word treatment

# In[20]:


chats_dict = {
    "AFAIK": "As Far As I Know",
    "AFK": "Away From Keyboard",
    "ASAP": "As Soon As Possible",
    "ATK": "At The Keyboard",
    "ATM": "At The Moment",
    "A3": "Anytime, Anywhere, Anyplace",
    "BAK": "Back At Keyboard",
    "BBL": "Be Back Later",
    "BBS": "Be Back Soon",
    "BFN": "Bye For Now",
    "B4N": "Bye For Now",
    "BRB": "Be Right Back",
    "BRT": "Be Right There",
    "BTW": "By The Way",
    "B4": "Before",
    "B4N": "Bye For Now",
    "CU": "See You",
    "CUL8R": "See You Later",
    "CYA": "See You",
    "FAQ": "Frequently Asked Questions",
    "FC": "Fingers Crossed",
    "FWIW": "For What It's Worth",
    "FYI": "For Your Information",
    "GAL": "Get A Life",
    "GG": "Good Game",
    "GN": "Good Night",
    "GMTA": "Great Minds Think Alike",
    "GR8": "Great!",
    "G9": "Genius",
    "IC": "I See",
    "ICQ": "I Seek you (also a chat program)",
    "ILU": "I Love You",
    "IMHO": "In My Honest/Humble Opinion",
    "IMO": "In My Opinion",
    "IOW": "In Other Words",
    "IRL": "In Real Life",
    "KISS": "Keep It Simple, Stupid",
    "LDR": "Long Distance Relationship",
    "LMAO": "Laugh My A.. Off",
    "LOL": "Laughing Out Loud",
    "LTNS": "Long Time No See",
    "L8R": "Later",
    "MTE": "My Thoughts Exactly",
    "M8": "Mate",
    "NRN": "No Reply Necessary",
    "OIC": "Oh I See",
    "PITA": "Pain In The A..",
    "PRT": "Party",
    "PRW": "Parents Are Watching",
    "QPSA": "Que Pasa?",
    "ROFL": "Rolling On The Floor Laughing",
    "ROFLOL": "Rolling On The Floor Laughing Out Loud",
    "ROTFLMAO": "Rolling On The Floor Laughing My A.. Off",
    "SK8": "Skate",
    "STATS": "Your sex and age",
    "ASL": "Age, Sex, Location",
    "THX": "Thank You",
    "TTFN": "Ta-Ta For Now!",
    "TTYL": "Talk To You Later",
    "U": "You",
    "U2": "You Too",
    "U4E": "Yours For Ever",
    "WB": "Welcome Back",
    "WTF": "What The F...",
    "WTG": "Way To Go!",
    "WUF": "Where Are You From?",
    "W8": "Wait...",
    "7K": "Sick:-D Laugher",
    "TFW": "That feeling when",
    "MFW": "My face when",
    "MRW": "My reaction when",
    "IFYP": "I feel your pain",
    "LOL": "Laughing out loud",
    "TNTL": "Trying not to laugh",
    "JK": "Just kidding",
    "IDC": "I don’t care",
    "ILY": "I love you",
    "IMU": "I miss you",
    "ADIH": "Another day in hell",
    "ZZZ": "Sleeping, bored, tired",
    "WYWH": "Wish you were here",
    "TIME": "Tears in my eyes",
    "BAE": "Before anyone else",
    "FIMH": "Forever in my heart",
    "BSAAW": "Big smile and a wink",
    "BWL": "Bursting with laughter",
    "LMAO": "Laughing my a** off",
    "BFF": "Best friends forever",
    "CSL": "Can’t stop laughing"
}


# In[21]:


def chat_conversion(text):
    new_text=[]
    for w in text.split():
        if w.upper() in chats_dict:
            new_text.append(chats_dict[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)


# In[22]:


chat_conversion('CSL on Ramesh')


# # 6. Spelling Correction

# In[23]:


from textblob import TextBlob


# In[24]:


incorrect_text = 'ceertain coonditon dduuring seveal ggenerattion'


# In[25]:


textBlb = TextBlob(incorrect_text)
textBlb.correct().string


# # 7. Removing Stop words
Stop words are common words in a language that are often filtered out during natural language processing tasks, such as 
text analysis and information retrieval. These words typically do not carry significant meaning or contribute to the 
understanding of the text. Examples of stop words in English include "the," "is," "at," "on," "in," and "of." Removing stop 
words can help improve the efficiency and accuracy of text processing tasks by reducing the amount of data to be analyzed and 
focusing on the more meaningful content.
# In[26]:


from nltk.corpus import stopwords


# In[27]:


stopwords.words('english')


# In[28]:


def remove_stopwords(text):
    new_text = []
    for word in text.split():
        if word in stopwords.words('english'):
            new_text.append('')
        else:
            new_text.append(word)
    x = new_text[:]
    new_text.clear()
    return " ".join(x)


# In[29]:


remove_stopwords('probably my all-time favourite movie, a story of selflessness and dedication to a noble cause,')


# # 8. Handling Emojis

# In[30]:


text = 'I am on🔥'


# In[31]:


import emoji
print(emoji.demojize(text))


# # 9. Tokenization

# In[32]:


# word tokenization
s1 = 'I am going to bengaluru.'
s1.split()


# In[33]:


# sentence tokenization
s2 = 'I am going to bengaluru. I am eating mango.'
s2.split('.')


# # Challenges in tokenization

# In[34]:


s3 = 'Where do think I should go? I have 3 days holiday'
s3.split('.')


# # Using NLTK library to do tokenization

# In[35]:


from nltk.tokenize import word_tokenize, sent_tokenize


# In[36]:


s = "I am going to visit delhi!"
word_tokenize(s)


# In[37]:


s = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed consequat felis eget justo vestibulum, sit amet lacinia turpis venenatis. Nunc vitae justo nec nisi sagittis blandit. Nullam at ex in justo cursus condimentum. Aliquam erat volutpat. Nulla facilisi. Proin accumsan sem sit amet varius mattis."


# In[38]:


sent_tokenize(s)


# In[39]:


s1 = 'I have a Ph.D. in A.I.'
s2 = "we're here to help! mail us at abc@gmail.com"
s3 = 'A 5km ride cost $10.50'


# In[40]:


word_tokenize(s1)


# In[41]:


word_tokenize(s2)


# In[42]:


word_tokenize(s3)


# # Using spacy library to do tokenization

# In[ ]:


import spacy
nlp = spacy.load('en_core_web_sm')


# In[ ]:


doc1 = nlp(s1)
doc2 = nlp(s2)
doc3 = nlp(s3)


# In[ ]:


for token in doc1:
    print(token)


# In[ ]:


for token in doc2:
    print(token)


# In[ ]:


for token in doc3:
    print(token)


# # 10. Stemming

# Stemming is the process of reducing words to their root or base form, even if the root itself may not be a valid word. 
# This is typically achieved by removing suffixes or prefixes from words, allowing variations of a word to be treated as a 
# single entity. For example, stemming would convert "running," "ran," and "runner" to the common root "run." It's commonly 
# used in natural language processing and text mining to simplify text analysis by treating different forms of the same word 
# as equivalent.

# In[43]:


# Useful in Information Retriever system like google search


# In[44]:


from nltk.stem.porter import PorterStemmer


# In[45]:


ps = PorterStemmer()
def stem_words(text):
    return " ".join([ps.stem(word) for word in text.split()])


# In[46]:


sample = "walk walks walked walking"
stem_words(sample)


# # 11. Lemmatization

# Lemmatization is the process of reducing words to their base or dictionary form, known as the lemma. Unlike stemming, 
# which simply chops off prefixes or suffixes, lemmatization considers the meaning of the word and applies morphological 
# analysis to transform it into its canonical form. This ensures that the resulting lemma is a valid word. For example, the 
# lemma of "was" is "be," and the lemma of "mice" is "mouse." Lemmatization is often preferred over stemming for tasks where 
# the meaning of words is important, such as in natural language understanding and information retrieval.

# Consider the word "better."
# 
# Stemming might simply remove suffixes to get the root word, so "better" might become "bet."
# 
# Lemmatization, however, considers the meaning of the word and reduces it to its base form or lemma. In this case, "better" 
# remains "better" because it's already in its base form.
# 
# 

# Stemming is generally faster than lemmatization because it's a simpler process. Stemming typically involves removing 
# common prefixes and suffixes from words to get to their root form, while lemmatization requires more complex morphological 
# analysis to return the base or dictionary form of a word.

# In[47]:


from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Example words
words = ["running", "ate", "beautiful", "cats"]

# Lemmatization
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
print("Lemmatized words:", lemmatized_words)


# In[48]:


from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Example words
words = ["running", "ate", "beautiful", "cats"]

# Lemmatization
lemmatized_words = [lemmatizer.lemmatize(word, pos='v') for word in words] # lemmatization based on verb
print("Lemmatized words:", lemmatized_words)

