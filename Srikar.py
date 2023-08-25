import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import cmudict
nltk.download('punkt')
nltk.download('cmudict')
import textstat

df = pd.read_excel('Input.xlsx')
df.head()

df1=pd.DataFrame()

stop_words = []
stop_words_files=['StopWords\\StopWords_Auditor.txt','StopWords\\StopWords_Currencies.txt','StopWords\\StopWords_DatesandNumbers.txt','StopWords\\StopWords_Generic.txt','StopWords\\StopWords_GenericLong.txt','StopWords\\StopWords_Geographic.txt','StopWords\\StopWords_Names.txt']
for file in stop_words_files:
    with open(file, 'r') as f:
        stop_words += f.read().splitlines()

positive_words = {}
negative_words = {}
with open('MasterDictionary\\positive-words.txt', 'r') as f:
    for line in f:
        if line not in stop_words:
            positive_words[line.strip()] = 1
with open(r'MasterDictionary\\negative-words.txt', 'r') as f:
    for line in f:
        if line not in stop_words:
            negative_words[line.strip()] = 1

df['POSITIVE SCORE'] = None
df['NEGATIVE SCORE'] = None
df['POLARITY SCORE'] = None
df['SUBJECTIVITY SCORE'] = None
df['AVG SENTENCE LENGTH'] = None
df['PERCENTAGE OF COMPLEX WORDS'] = None
df['FOG INDEX'] = None
df['AVG NUMBER OF WORDS PER SENTENCE']=None
df['COMPLEX WORD COUNT']=None
df['WORD COUNT']=None
df['SYLLABLE PER WORD']=None
df['PERSONAL PRONOUNS']=None
df['AVG WORD LENGTH']=None

with open('URL_ID.txt', 'w', encoding='utf-8') as f:
    for index, row in df.iterrows():
        # Get the URL from the current row
        url = row['URL']

        # Make a request to the URL and get the HTML content
        response = requests.get(url, timeout=100)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract the article title and content from the HTML content
        title = soup.find('title').text
        content = ''
        for paragraph in soup.find_all('p'):
            content += paragraph.text + '\n'

        #write the content to a new dataframe
        df1.at[index, 'URL'] = url
        df1.at[index, 'title'] = title
        df1.at[index, 'content'] = content

        # Write the extracted content to the text file
        f.write(content)
        f.write('\n\n')  # Add a separator between articles

        # Clean the text by removing stop words
        tokens = word_tokenize(content.lower())
        filtered_tokens = [token for token in tokens if token not in stop_words]
        cleaned_content = ' '.join(filtered_tokens)
        total_words = len(filtered_tokens)

        # Perform sentiment analysis on the article content
        positive_score = sum([positive_words.get(word, 0) for word in cleaned_content.split()])
        negative_score = sum([negative_words.get(word, 0) for word in cleaned_content.split()])
        polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
        subjectivity_score = (positive_score + negative_score) / ((total_words) + 0.000001)
        personal_pronouns_list = ['I', 'we', 'my', 'ours', 'us']

        # Analysis of Readability
        sentences = sent_tokenize(content)
        total_sentences = len(sentences)
        total_words = len(filtered_tokens)
        complex_words = 0
        total_syllables = 0
        personal_pronouns = 0
        for word in filtered_tokens:
            # Count syllables in each word
            syllables = textstat.syllable_count(word)
            total_syllables += syllables

            # Check if the word is a complex word
            if syllables > 2:
                complex_words += 1

            # Check if the word is a personal pronoun
            if word in personal_pronouns_list:
                personal_pronouns += 1

        # Calculate the average sentence length
        avg_sentence_length = total_words / total_sentences

        # Calculate the percentage of complex words
        percent_complex_words = (complex_words / total_words) * 100

        # Calculate the Fog Index
        fog_index = 0.4 * (avg_sentence_length + percent_complex_words)

        # Calculate the average number of words per sentence
        avg_words_per_sentence = total_words / total_sentences

        # Calculate the average word length
        total_characters = sum(len(word) for word in filtered_tokens)
        avg_word_length = total_characters / total_words

        # Update the dataframe with the sentiment analysis results
        df.at[index, 'POSITIVE SCORE'] = positive_score
        df.at[index, 'NEGATIVE SCORE'] = negative_score
        df.at[index, 'POLARITY SCORE'] = polarity_score
        df.at[index, 'SUBJECTIVITY SCORE'] = subjectivity_score
        df.at[index, 'AVG SENTENCE LENGTH'] = avg_sentence_length
        df.at[index, 'PERCENTAGE OF COMPLEX WORDS'] = percent_complex_words
        df.at[index, 'FOG INDEX'] = fog_index
        df.at[index, 'AVG NUMBER OF WORDS PER SENTENCE'] = avg_words_per_sentence
        df.at[index, 'COMPLEX WORD COUNT'] = complex_words
        df.at[index, 'WORD COUNT'] = total_words
        df.at[index, 'SYLLABLE PER WORD'] = total_syllables / total_words
        df.at[index, 'PERSONAL PRONOUNS'] = personal_pronouns
        df.at[index, 'AVG WORD LENGTH'] = avg_word_length

df.head()

df1.head()

#write this df to xslx format
df.to_excel('Output Data Structure.xlsx', index=False)

