import streamlit as st 
import pandas as pd
from PIL import Image
import spacy
import string
import numpy as np
from gensim.summarization import summarize
import yake
import plotly.express as px

import re
import nltk
import unidecode
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('wordnet') 
from os import path
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns


# function to extract entities
@st.cache(suppress_st_warning = True)
def entity_analyzer(text, lang_model):
	nlp = spacy.load(lang_model)
	doc = nlp(text)
	tokens = [token.text for token in doc]
	entities = [(entity.text, entity.label_) for entity in doc.ents]
	return ['Entities":{}'.format(entities)]

# function for anonymization
# @st.cache(suppress_st_warning = True)
def sanitize_names(text, lang_model):
	nlp = spacy.load(lang_model)
	doc = nlp(text)
	redacted_sentences = []
	for ent in doc.ents:
		ent.merge()
	for token in doc:
		if token.ent_type_ in ['PER', 'PERSON']:
			redacted_sentences.append("[CCCCCCENSOREDDDDDDD] ")
		else:
			redacted_sentences.append(token.string)
	return "".join(redacted_sentences)	

@st.cache(suppress_st_warning = True)
def replace_punctuation(text):
	new_text_ls = []
	text_ls = [char for char in text]
	for char in text_ls:
		if char in string.punctuation:
			char = " "

		new_text_ls.append(char)

	return "".join(new_text_ls)

@st.cache(suppress_st_warning = True)
def clean_string(text, lang_model):
	text = str(text)
	text = text.lower()
	text = replace_punctuation(text)
	text = [word for word in text.split(" ")]
	text = [word for word in text if not any(c.isdigit() for c in word)]
	stop_words = set(stopwords.words(lang_model))
	text = [tok for tok in text if not tok in stop_words]
	text = " ".join(text).strip()
	text = unidecode.unidecode(text)
	return text
	
@st.cache(suppress_st_warning = True)
def get_top_n_words(corpus, ngrams, n=None):
	vec1 = CountVectorizer(ngram_range=(ngrams, ngrams), 
		max_features=2000).fit(corpus)
	bag_of_words = vec1.transform(corpus)
	sum_words = bag_of_words.sum(axis=0) 
	words_freq = [(word, sum_words[0, idx]) for word, idx in     
				vec1.vocabulary_.items()]
	words_freq =sorted(words_freq, key = lambda x: x[1], 
				reverse=True)
	return words_freq[:n]


def main():

	image = Image.open('images/wordcloud.png')

	st.sidebar.image(image,  width=200)
	st.sidebar.header("NLP demos")
	st.sidebar.text("Select an option and see it in action!")

	st.title("Natural Language Processing demos")
	st.markdown("""
    	#### An NLP app for demonstration purposes: analyze your text!
    	

    	""")


	# Named Entity Recognition

	if st.sidebar.checkbox("Named Entity Recognition"):

		lang_options = st.selectbox("Choose language (EN/PT)",['EN','PT'])

		if lang_options == 'EN':
			lang_model = 'en_core_web_sm'
		elif lang_options == 'PT':
			lang_model = 'pt_core_news_sm'
		else:
			lang_model = 'en_core_web_sm'

		message = st.text_area("Enter text inside the box...")

		if st.button("Run"):
			with st.spinner('Wait for it...'):
				entity_result = entity_analyzer(message, lang_model)
			st.success(st.json(entity_result))


	# Summarization

	if st.sidebar.checkbox("Text Summarization"):
		st.subheader("Summarize Your Text")

		message = st.text_area("Enter text (EN only for now) inside the box...")

		ratio_value = st.slider('Select a ratio (%) that determines the proportion of the number of sentences of the original text to be chosen for the summary', 0, 100, (10))

		if st.button("Run"):
			with st.spinner('Wait for it...'):
				summary_result = summarize(message, ratio=ratio_value/100)
			st.success(summary_result)


	# Automated Keyword Extraction

	if st.sidebar.checkbox("Automated Keyword Extraction"):
		st.subheader("Extract Keywords")

		lang_options = st.selectbox("Choose language (EN/PT)",['EN','PT'])

		if lang_options == 'EN':
			lang_model = 'en'
		elif lang_options == 'PT':
			lang_model = 'pt'
		else:
			lang_model = 'en'

		message = st.text_area("Enter text inside the box...")

		if st.button("Run"):
			with st.spinner('Wait for it...'):
				
				# set YAKE! parameters
				language = lang_model
				max_ngram_size = 2
				deduplication_thresold = 0.2
				deduplication_algo = "seqm"
				windowSize = 1
				numOfKeywords = 10

				custom_kw_extractor = yake.KeywordExtractor(
					lan=language,
					n=max_ngram_size,
					dedupLim=deduplication_thresold,
					dedupFunc=deduplication_algo,
					windowsSize=windowSize,
					top=numOfKeywords,
					features=None,
				)
				keywords = custom_kw_extractor.extract_keywords(message)
				keywords = [kw for kw, res in keywords]
				
				st.success('Keywords: ' + (', '.join(sorted(keywords))))


	# Data Anonymization (erasing names)

	if st.sidebar.checkbox("Anonymize Personal Data"):
		st.subheader("Anonymize Your Data: Hiding Names")

		lang_options = st.selectbox("Choose language (EN/PT)",['EN','PT'])

		if lang_options == 'EN':
			lang_model = 'en_core_web_sm'
		elif lang_options == 'PT':
			lang_model = 'pt_core_news_sm'
		else:
			lang_model = 'en_core_web_sm'

		message = st.text_area("Enter text inside the box...")

		if st.button("Run"):
			with st.spinner('Wait for it...'):
				names_cleaned_result = sanitize_names(message, lang_model)
				st.success(names_cleaned_result)



	# N-grams

	if st.sidebar.checkbox("N-Grams Barplot"):
		st.subheader("Visualize an N-grams barplot")

		lang_options = st.selectbox("Choose language (EN/PT)",['EN','PT'])

		if lang_options == 'EN':
			lang_model = 'english'
		elif lang_options == 'PT':
			lang_model = 'portuguese'
		else:
			lang_model = 'english'

		ngram_options = st.selectbox("Choose N for N-grams (1, 2 or 3)",[1,2,3])

		if ngram_options == 1:
			ngrams = 1
		elif ngram_options == 2:
			ngrams = 2
		else:
			ngrams = 3

		message = st.text_area("Let's analyze and get some visuals...")
		
		if st.button("Run"):
			with st.spinner('Wait for it...'):
				corpus = []
				
				text = ''.join([unidecode.unidecode(accented_string) for accented_string in message])
				
				corpus.append(clean_string(text, lang_model))

				top3_words = get_top_n_words(corpus, ngrams, n=20)
				top3_df = pd.DataFrame(top3_words)
				top3_df.columns=["N-gram", "Freq"]
				fig = px.bar(top3_df, x='N-gram', y='Freq')
				
				st.plotly_chart(fig)


	# Wordcloud

	if st.sidebar.checkbox("Wordcloud"):
		st.subheader("Visualize a wordcloud")

		lang_options = st.selectbox("Choose language (EN/PT)",['EN','PT'])

		if lang_options == 'EN':
			lang_model = 'english'
		elif lang_options == 'PT':
			lang_model = 'portuguese'
		else:
			lang_model = 'english'

		message = st.text_area("Let's analyze and get some visuals...")
		
		if st.button("Run"):
			with st.spinner('Wait for it...'):
				corpus = []
				
				text = ''.join([unidecode.unidecode(accented_string) for accented_string in message])
				
				corpus.append(clean_string(text, lang_model))
				
				stop_words = set(stopwords.words(lang_model))

				#Word cloud
				wordcloud = WordCloud(
										background_color='white',
										stopwords=stop_words,
										max_words=100,
										max_font_size=50, 
										random_state=42
										).generate(str(corpus))
				fig = plt.figure(1)
				plt.imshow(wordcloud, interpolation="bilinear")
				plt.axis('off')
				st.pyplot()



if __name__ == '__main__':
	main()
