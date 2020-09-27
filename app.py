import streamlit as st 
import pandas as pd
from PIL import Image
import spacy
import string
import numpy as np
from gensim.summarization import summarize
from spacy.lang.en.stop_words import STOP_WORDS as en_stopwords
from spacy.lang.pt.stop_words import STOP_WORDS as pt_stopwords
from collections import OrderedDict
# import yake
import plotly.express as px

import re
# import nltk
import unidecode
# nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
# nltk.download('wordnet') 
from os import path
from PIL import Image
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# adding non accented words in pt_stopwords
pt_stopwords = pt_stopwords.union(set([unidecode.unidecode(k) for k in pt_stopwords]))


# text rank class

@st.cache(suppress_st_warning = True, allow_output_mutation=True)
class TextRank4Keyword():
    """Extract keywords from text"""
    
    def __init__(self):
        self.d = 0.85 # damping coefficient, usually is .85
        self.min_diff = 1e-5 # convergence threshold
        self.steps = 10 # iteration steps
        self.node_weight = None # save keywords and its weight

    
    def set_stopwords(self, stopwords, lang_model):  
        """Set stop words"""
        nlp = spacy.load(lang_model)
        for word in stopwords:
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True
    
    def sentence_segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences
        
    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab
    
    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs
        
    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())
    
    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1
            
        # Get Symmeric matrix
        g = self.symmetrize(g)
        
        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm!=0) # this is ignore the 0 element in norm
        
        return g_norm

    
    # def get_keywords(self, number=10):
    #     """Print top number keywords"""
    #     node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
    #     for i, (key, value) in enumerate(node_weight.items()):
    #         print(key + ' - ' + str(value))
    #         if i > number:
    #             break

    def get_keywords(self, number=10):
        """Print top number keywords"""
        keywords = []
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        for i, (key, value) in enumerate(node_weight.items()):
            # print(key + ' - ' + str(value))
            keywords.append(key)
            if i > number:
                break
        return keywords
        
        
    def analyze(self, text, lang_model,
                candidate_pos=['NOUN', 'PROPN', 'VERB'], 
                window_size=4, lower=False, stopwords=list()):
        """Main function to analyze text"""
        
        # Set stop words
        self.set_stopwords(stopwords, lang_model)
        
        # Pare text by spaCy
        nlp = spacy.load(lang_model)
        doc = nlp(text)
        
        # Filter sentences
        sentences = self.sentence_segment(doc, candidate_pos, lower) # list of list of words
        
        # Build vocabulary
        vocab = self.get_vocab(sentences)
        
        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)
        
        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)
        
        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))
        
        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1-self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr))  < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]
        
        self.node_weight = node_weight


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
    text_ls = [char for char in text]
    text_ls = [t.replace("\n", " ") for t in text_ls]
    text_ls = [char for char in text_ls if char.isalnum() or char == " "]
    text_ls = "".join(text_ls)
    
    return "".join(text_ls)

@st.cache(suppress_st_warning = True)
def clean_string(text, lang_options):
    text = str(text)
    text = text.lower()
    text = replace_punctuation(text)
    text = [word for word in text.split(" ")]
    text = [word for word in text if not any(c.isdigit() for c in word)]
    if lang_options == 'EN':
        stop_words = en_stopwords
    else:
        stop_words = pt_stopwords
    text = [tok for tok in text if not tok in stop_words and tok != '']
    text = " ".join(text)
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
    words_freq = sorted(words_freq, key = lambda x: x[1], 
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

    if st.sidebar.checkbox("Named Entity Recognition", key='check1'):

        lang_options = st.selectbox("Choose language (EN/PT)",['EN','PT'], key='sel1')

        if lang_options == 'EN':
            lang_model = 'en_core_web_sm'
        else:
            lang_model = 'pt_core_news_sm'

        message = st.text_area("Enter text inside the box...", key='ins1')

        if st.button("Run", key='run1'):
            with st.spinner('Wait for it...'):
                entity_result = entity_analyzer(message, lang_model)
            st.success(st.json(entity_result))


    # Summarization

    if st.sidebar.checkbox("Text Summarization", key='check2'):
        st.subheader("Summarize Your Text")

        message = st.text_area("Enter text (EN only for now) inside the box...", key='ins2')

        ratio_value = st.slider('Select a ratio (%) that determines the proportion of the number of sentences of the original text to be chosen for the summary', 0, 100, (10))

        if st.button("Run", key='run2'):
            with st.spinner('Wait for it...'):
                summary_result = summarize(message, ratio=ratio_value/100)
            st.success(summary_result)


    # # Automated Keyword Extraction

    # if st.sidebar.checkbox("Automated Keyword Extraction"):
    # 	st.subheader("Extract Keywords")

    # 	lang_options = st.selectbox("Choose language (EN/PT)",['EN','PT'])

    # 	if lang_options == 'EN':
    # 		lang_model = 'en'
    # 	elif lang_options == 'PT':
    # 		lang_model = 'pt'
    # 	else:
    # 		lang_model = 'en'

    # 	message = st.text_area("Enter text inside the box...")

    # 	if st.button("Run"):
    # 		with st.spinner('Wait for it...'):
                
    # 			# set YAKE! parameters
    # 			language = lang_model
    # 			max_ngram_size = 2
    # 			deduplication_thresold = 0.2
    # 			deduplication_algo = "seqm"
    # 			windowSize = 1
    # 			numOfKeywords = 10

    # 			custom_kw_extractor = yake.KeywordExtractor(
    # 				lan=language,
    # 				n=max_ngram_size,
    # 				dedupLim=deduplication_thresold,
    # 				dedupFunc=deduplication_algo,
    # 				windowsSize=windowSize,
    # 				top=numOfKeywords,
    # 				features=None,
    # 			)
    # 			keywords = custom_kw_extractor.extract_keywords(message)
    # 			keywords = [kw for kw, res in keywords]
                
    # 			st.success('Keywords: ' + (', '.join(sorted(keywords))))


# Automated Keyword Extraction

    if st.sidebar.checkbox("Automated Keyword Extraction", key='check3'):
        st.subheader("Extract Keywords")

        lang_options = st.selectbox("Choose language (EN/PT)",['EN','PT'], key='sel2')

        if lang_options == 'EN':
            stop_words = en_stopwords
            lang_model = 'en_core_web_sm'
        else:
            lang_model = 'pt_core_news_sm'
            stop_words = pt_stopwords

        # nlp = spacy.load(lang_model)


        message = st.text_area("Enter text inside the box...", key='ins3')

        if st.button("Run", key='run3'):
            with st.spinner('Wait for it...'):
                
                # corpus = []
                
                text = ''.join([unidecode.unidecode(accented_string) for accented_string in message])
                
                corpus = clean_string(text, lang_options)

                tr4w = TextRank4Keyword()
                tr4w.set_stopwords(stopwords=stop_words, lang_model=lang_model)
                # tr4w.set_stopwords(stopwords=stop_words)
                # tr4w.analyze(ppp, candidate_pos = ['NOUN', 'PROPN', 'VERB'], window_size=4, lower=False)
                tr4w.analyze(corpus, window_size=4, lower=False, lang_model=lang_model)

                st.success('Keywords: ' + (', '.join(sorted(tr4w.get_keywords(10)))))



    # Data Anonymization (erasing names)

    if st.sidebar.checkbox("Anonymize Personal Data"):
        st.subheader("Anonymize Your Data: Hiding Names")

        lang_options = st.selectbox("Choose language (EN/PT)",['EN','PT'], key='sel3')

        if lang_options == 'EN':
            lang_model = 'en_core_web_sm'
        elif lang_options == 'PT':
            lang_model = 'pt_core_news_sm'
        else:
            lang_model = 'en_core_web_sm'

        message = st.text_area("Enter text inside the box...", key='ins4')

        if st.button("Run", key='run4'):
            with st.spinner('Wait for it...'):
                names_cleaned_result = sanitize_names(message, lang_model)
                st.success(names_cleaned_result)



    # N-grams

    if st.sidebar.checkbox("N-Grams Barplot"):
        st.subheader("Visualize an N-grams barplot")

        lang_option = st.selectbox("Choose language (EN/PT)",['EN','PT'], key='sel4')

        # if lang_options == 'EN':
        #     lang_model = 'english'
        # elif lang_options == 'PT':
        #     lang_model = 'portuguese'
        # else:
        #     lang_model = 'english'

        ngram_option = st.selectbox("Choose N for N-grams (1, 2 or 3)",[1,2,3], key='sel5')

        # if ngram_options == 1:
        #     ngrams = 1
        # elif ngram_options == 2:
        #     ngrams = 2
        # else:
        #     ngrams = 3

        message = st.text_area("Let's analyze and get some visuals...", key='ins5')
        
        if st.button("Run", key='run5'):
            with st.spinner('Wait for it...'):
                corpus = []
                
                text = ''.join([unidecode.unidecode(accented_string) for accented_string in message])
                
                corpus.append(clean_string(text, lang_option))

                top3_words = get_top_n_words(corpus, ngram_option, n=20)
                top3_df = pd.DataFrame(top3_words)
                top3_df.columns=["N-gram", "Freq"]
                fig = px.bar(top3_df, x='N-gram', y='Freq')
                
                st.plotly_chart(fig)


    # Wordcloud

    if st.sidebar.checkbox("Wordcloud"):
        st.subheader("Visualize a wordcloud")

        lang_option = st.selectbox("Choose language (EN/PT)",['EN','PT'], key='sel6')

        if lang_option == 'EN':
            # lang_model = 'en_core_web_sm'
            stop_words = en_stopwords
        else:
            # lang_model = 'pt_core_news_sm'
            stop_words = pt_stopwords

        message = st.text_area("Let's analyze and get some visuals...", key='ins6')
        
        if st.button("Run", key='run6'):
            with st.spinner('Wait for it...'):
                corpus = []
                
                text = ''.join([unidecode.unidecode(accented_string) for accented_string in message])
                
                corpus.append(clean_string(text, lang_option))
                
                
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
