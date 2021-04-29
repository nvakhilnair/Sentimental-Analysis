import requests
from bs4 import BeautifulSoup as bs
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

rev =[]
for i in range(1,11,1):
    url="https://www.flipkart.com/poco-m2-pitch-black-64-gb/product-reviews/itm630300707d0a5?pid=MOBFV9V92KHMFCVF&lid=LSTMOBFV9V92KHMFCVFEXYXG8&sortOrder=MOST_HELPFUL&certifiedBuyer=false&aid=overall&page=" + str(i)
    response = requests.get(url)
    soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
    reviews = soup.find_all("div",attrs={"class","t-ZTKy"})# Extracting the content under specific tags  
    for j in range(len(reviews)):
        rev.append(reviews[j].text)

rev_string = " ".join(rev)
rev_string = re.sub("[^A-Za-z" "]+"," ", rev_string).lower()
rev_string = re.sub("[0-9" "]+"," ", rev_string)
reviews_words = rev_string.split(" ")



vectorizer = TfidfVectorizer(reviews_words, use_idf=True,ngram_range=(1, 3))
X = vectorizer.fit_transform(reviews_words)


with open("/home/akhilnv/Desktop/Data Science/Assignment/text mining/common word.txt","r") as sw:
    stop_words = sw.read()
    
stop_words = stop_words.split("\n")
stop_words.extend(["pocom2","m2","time","android","phone","device","screen","battery","product","good","day","price",'flipkart'])
reviews_words = [w for w in reviews_words if not w in stop_words]
rev_string = " ".join(reviews_words)


wordcloud = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(rev_string)

plt.imshow(wordcloud)


with open("/home/akhilnv/Desktop/Data Science/text mining/Datasets NLP/pos.txt","r") as pos:
  poswords = pos.read().split("\n")

ip_pos_in_pos = " ".join ([w for w in reviews_words if w in poswords])

wordcloud_pos_in_pos = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_pos_in_pos)
plt.figure(2)
plt.imshow(wordcloud_pos_in_pos)


with open("/home/akhilnv/Desktop/Data Science/text mining/Datasets NLP/neg.txt", "r") as neg:
  negwords = neg.read().split("\n")

ip_neg_in_neg = " ".join ([w for w in reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_neg_in_neg)
plt.figure(3)
plt.imshow(wordcloud_neg_in_neg)




# wordcloud with bigram
nltk.download('punkt')
from wordcloud import WordCloud, STOPWORDS

WNL = nltk.WordNetLemmatizer()

# Lowercase and tokenize
text = rev_string.lower()

# Remove single quote early since it causes problems with the tokenizer.
text = text.replace("'", "")

tokens = nltk.word_tokenize(text)
text1 = nltk.Text(tokens)

# Remove extra chars and remove stop words.
text_content = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text1]

# Create a set of stopwords
stopwords_wc = set(STOPWORDS)
customised_words = ['price', 'great'] # If you want to remove any particular word form text which does not contribute much in meaning

new_stopwords = stopwords_wc.union(customised_words)

# Remove stop words
text_content = [word for word in text_content if word not in new_stopwords]

# Take only non-empty entries
text_content = [s for s in text_content if len(s) != 0]


nltk_tokens = nltk.word_tokenize(text)  
bigrams_list = list(nltk.bigrams(text_content))
print(bigrams_list)

dictionary2 = [' '.join(tup) for tup in bigrams_list]
print (dictionary2)

# Using count vectoriser to view the frequency of bigrams
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(2, 2))
bag_of_words = vectorizer.fit_transform(dictionary2)
vectorizer.vocabulary_

sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
print(words_freq[:100])

# Generating wordcloud
words_dict = dict(words_freq)
WC_height = 1000
WC_width = 1500
WC_max_words = 200
wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width, stopwords=new_stopwords)
wordCloud.generate_from_frequencies(words_dict)

plt.figure(4)
plt.title('Most frequently occurring bigrams connected by same colour and font size')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()


