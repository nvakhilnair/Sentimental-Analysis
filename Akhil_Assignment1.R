# importing the required libraries
library(base64enc)
library(httpuv)
library(rvest)
library(XML)
library(magrittr)
library(tm)
library(wordcloud)
library(wordcloud2)
library(RWeka)
library(stringr)
library(readr)

#adding custom wordlist to improve the model
common_word <- read_csv("common word.txt", col_names = FALSE)
Cwords <- as.list(common_word)
Cwords <- as.vector(Cwords[['X1']])


# url for the website from data extraction
aurl <- "https://www.flipkart.com/poco-m2-pitch-black-64-gb/product-reviews/itm630300707d0a5?pid=MOBFV9V92KHMFCVF&lid=LSTMOBFV9V92KHMFCVFEXYXG8&sortOrder=MOST_HELPFUL&certifiedBuyer=false&aid=overall&page"

#creating the variable to store the reviews
flipkart_reviews <- NULL

#looping through each page of reviews and extracting it
for (i in 1:10){
  murl <- read_html(as.character(paste(aurl,i,sep="=")))
  rev <- murl %>% html_nodes(".t-ZTKy") %>% html_text()
  #remove READ MORE word from each review extracted
  rev <- str_remove_all(rev, "READ MORE")
  flipkart_reviews <- c(flipkart_reviews,rev)
}

# Assign reviews to txt variable
txt <- flipkart_reviews

# creating corpus vector from data pre processign
x <- Corpus(VectorSource(txt))
x <- tm_map(x, function(x) iconv(enc2utf8(x), sub='byte'))
x1 <- tm_map(x, tolower)
x1 <- tm_map(x1, removePunctuation)
x1 <- tm_map(x1, removeNumbers)
x1 <- tm_map(x1, removeWords, stopwords('english'))
x1 <- tm_map(x1, removeWords, c('phone'))
x1 <- tm_map(x1, removeWords,Cwords)
x1 <- tm_map(x1, stripWhitespace)

#storing the pre processed data in a data frame for reference
data = data.frame(text = sapply(x1, as.character), stringsAsFactors = FALSE)

#Converting the reviews into TDM and DTM
tdm <- TermDocumentMatrix(x1)
dtm <- t(tdm) # transpose
dtm <- DocumentTermMatrix(x1)


#removing sparse terms
corpus.dtm.frequent <- removeSparseTerms(tdm, 0.2)

#converting as matrix
tdm <- as.matrix(tdm)

##################################### Sentimental Analysis ####################################

# Select Words which are occurring more than or equal to 9 times
w <- rowSums(tdm)
w_sub <- subset(w, w >= 9)

#Plotting the bar graph
barplot(w_sub, las=2, col = rainbow(5))

#plotting the word cloud
w_sub1 <- sort(rowSums(tdm), decreasing = F)
wordcloud(words = names(w_sub1), freq = w_sub1, random.order=F, colors=rainbow(10), scale = c(3,0.5), rot.per = 0.1)

#creating data frame with words and its frequency of occurrence
w1 <- data.frame(names(w_sub), w_sub)
colnames(w1) <- c('word', 'freq')

#Wordcloud in different shapes
wordcloud2(w1, size=0.5, shape='circle')
wordcloud2(w1, size=0.5, shape = 'triangle')
wordcloud2(w1, size=0.5, shape = 'star')

#Plotting Bigram
minfreq_bigram <- 2
bitoken <- NGramTokenizer(x1, Weka_control(min = 2, max = 2))
two_word <- data.frame(table(bitoken))
sort_two <- two_word[order(two_word$Freq, decreasing = T), ]
wordcloud(sort_two$bitoken, sort_two$Freq, random.order = F, scale = c(2, 0.2), min.freq = minfreq_bigram, colors = brewer.pal(8, "Dark2"), max.words =60)
