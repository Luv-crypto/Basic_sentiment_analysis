# importing libraries


import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import string
from nltk.tokenize import sent_tokenize, word_tokenize , RegexpTokenizer
from nltk.corpus import cmudict
import syllapy
import re

# To help website recognize us
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
PC  = pd.read_excel("input(1).xlsx")


# for main looping the code
Links = PC["URL"].to_list()
url_id = PC["URL_ID"].to_list()



# count syllables
def syllable_count(word):
    try:
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
    except KeyError:
        return syllapy.count(word)


# To store all the scores of all articles

positive_scores = []
negative_scores = []
Polarity_Scores = []
Subjectivity_scores = []
Average_Sentence_Lengths = []
Percentage_comp_words = []
Fog_Indices = []
complex_word_counts = []
word_counts = []
syll_per_words = []
pronouns_counts = []
avg_word_lengths = []



# Main loop

for i in range(len(Links)):
    url= Links[i]
    res = requests.get(url, headers=headers)
    html_page = res.content
    
    
# extracting title and paragraph from article

    soup = BeautifulSoup(html_page, 'html.parser')
    article_txts = []
    article = [
    title.getText(strip=True) for title in
    soup.find_all(["title", "p"])]
    txt_article = " ".join(article)
    article_txts.append(txt_article)
    

#  To save  the scraped text in txtfile

    UF= "url_"+ str(i+1) +".txt"
    with open(UF, "w",encoding = "utf-8-sig") as fp:
        fp.writelines(txt_article)
    
# Merging all the stopwords 

    filenames = ['StopWords_Auditor.txt', 'StopWords_Currencies.txt','StopWords_DatesandNumbers.txt','StopWords_Generic.txt','StopWords_GenericLong.txt','StopWords_Geographic.txt','StopWords_Names.txt']
    with open('final_stopwords.txt', 'w') as outfile:
        for names in filenames:
            with open(names) as infile:
                outfile.write(infile.read())
            outfile.write("\n")
            
            
 # getting the text after removing stopwords
          
    with open('final_stopwords.txt') as f:
        for line in f:
            words = line.split()
            if words:
                words[0]

    k = []
    with open('final_stopwords.txt', 'r') as f:
        for word in f:
            word = word.split('\n')
            k.append(word[0])
    p = [t for t in article_txts if t not in k]
    
    
  # using nlp to tokenize the words and sentences

    tokenizer = RegexpTokenizer(r'\w+')
    z= []
    for aa in p:
        z.append(tokenizer.tokenize(aa))
    [tokenized_words] = z
    
    
    r = []
    for bb in p:
        r.append(sent_tokenize(bb))
    [tokenized_sent] = r
    
    
    # Using master dictionary to hel calculate metrics
    
    positve_words=[cc.strip() for cc in open('positive-words.txt')]
    negative_words=[dd.strip() for dd in open('negative-words.txt')]
    
    
    positive_score = 0
    for ee in range(len(positve_words)):
        for ff in range(len(tokenized_words)):
            if positve_words[ee] == tokenized_words[ff]:
                positive_score+=1
    positive_scores.append(positive_score)
    
    
    negative_score = 0
    for gg in range(len(negative_words)):
        for hh in range(len(tokenized_words)):
            if negative_words[gg] == tokenized_words[hh]:
                negative_score+=1
    negative_scores.append(negative_score)
    
    
    Polarity_Score = (positive_score - negative_score)/ ((positive_score + negative_score) + 0.000001)
    Polarity_Scores.append(Polarity_Score)
    
    
    Subjectivity_score = (positive_score + negative_score)/((len(tokenized_words))+0.000001)
    Subjectivity_scores.append(Subjectivity_score)
    
    Average_Sentence_Length = len(tokenized_words)/len(tokenized_sent)
    Average_Sentence_Lengths.append(Average_Sentence_Length)
    
    d = cmudict.dict()
    
    complex_words = []
    for ii in range(len(tokenized_words)):
        if syllable_count(tokenized_words[ii]) >2:
            complex_words.append(tokenized_words[ii])
            
            
    Percentage_comp_word = len(complex_words)/len(tokenized_words)
    Percentage_comp_words.append(Percentage_comp_word)
    
    
    Fog_Index = 0.4 * (Average_Sentence_Length + len(complex_words))
    Fog_Indices.append(Fog_Index)
    
    complex_word_count = len(complex_words)
    complex_word_counts.append(complex_word_count)
    
    stop = set(stopwords.words('english'))
    word_count_nlp = [jj for jj in tokenized_words if jj not in stop and jj not in string.punctuation]
    sent_count_nlp = [kk for kk in tokenized_sent if kk not in stop and kk not in string.punctuation]
    
    
    word_count = len(word_count_nlp)
    word_counts.append(word_count)
    
    
    syll_per_word = 0 
    for ll in range(len(word_count_nlp)):
        a = syllable_count(word_count_nlp[ll]) 
        syll_per_word+=a
    syll_per_word = syll_per_word/word_count
    syll_per_words.append(syll_per_word)
    
    
    pronounRegex = re.compile(r'\b(I|we|my|ours|(?-i:us))\b',re.I)
    pronouns = pronounRegex.findall(p[0])
    pronouns_count = len(pronouns)
    if pronouns == []:
        pronouns_count = 0
    else:
        pass
    pronouns_counts.append(pronouns_count)
    
    
    words = p[0].split()
    wordCount = len(word_count_nlp)
    sum = 0

    for pp in words:
        ch = len(pp)
        sum = sum + ch
        avg_word_length = float(sum)/float(wordCount)
    avg_word_lengths.append(avg_word_length)


# Storing the data in excel

data = {"URL_ID":url_id,
    "URL":Links,
    "POSITIVE SCORE":positive_scores,
       "NEGATIVE SCORE": negative_scores,
       "POLARITY SCORE": Polarity_Scores,
       "SUBJECTIVITY SCORE":Subjectivity_scores,
       "AVG SENTENCE LENGTH": Average_Sentence_Lengths,
       "PERCENTAGE OF COMPLEX WORDS": Percentage_comp_words,
       "FOG INDEX":Fog_Indices,
       "AVG NUMBER OF WORDS PER SENTENCE" : Average_Sentence_Lengths,
       "COMPLEX WORD COUNT": complex_word_counts,
       "WORD COUNT":word_counts,
       "SYLLABLE PER WORD":syll_per_words,
       "PERSONAL PRONOUNS":pronouns_counts,
       "AVG WORD LENGTH":avg_word_lengths}
df = pd.DataFrame(data)
df.to_excel("Blackcoffer_output.xlsx", index=False)

