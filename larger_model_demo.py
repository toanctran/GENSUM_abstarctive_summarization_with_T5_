import streamlit as st
import transformers
import time
import tensorflow as tf
from transformers import T5Tokenizer, TFT5Model, TFT5ForConditionalGeneration

from spacy.lang.pt.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
import en_core_web_sm

st.set_page_config(layout="wide")
def summarization(text):
    
    nlp = en_core_web_sm.load()

    doc = nlp(text)


    corpus = [sent.text.lower() for sent in doc.sents ]
    
    cv = CountVectorizer(stop_words=list(STOP_WORDS))   
    cv_fit=cv.fit_transform(corpus)    
    word_list = cv.get_feature_names();    
    count_list = cv_fit.toarray().sum(axis=0)    

    word_frequency = dict(zip(word_list,count_list))

    val=sorted(word_frequency.values())

    # Check words with higher frequencies
    higher_word_frequencies = [word for word,freq in word_frequency.items() if freq in val[-3:]]
    print("\nWords with higher frequencies: ", higher_word_frequencies)

    # gets relative frequencies of words
    higher_frequency = val[-1]
    for word in word_frequency.keys():  
        word_frequency[word] = (word_frequency[word]/higher_frequency)


    # SENTENCE RANKING: the rank of sentences is based on the word frequencies
    sentence_rank={}
    for sent in doc.sents:
        for word in sent :       
            if word.text.lower() in word_frequency.keys():            
                if sent in sentence_rank.keys():
                    sentence_rank[sent]+=word_frequency[word.text.lower()]
                else:
                    sentence_rank[sent]=word_frequency[word.text.lower()]
            else:
                continue

    top_sentences=(sorted(sentence_rank.values())[::-1])
    top_sent=top_sentences[:5]

    # Mount summary
    summary=[]
    for sent,strength in sentence_rank.items():  
        if strength in top_sent:
            summary.append(sent)

    # return orinal text and summary
    return summary

st.title('ABSTRACTIVE TEXT SUMMARIZATION')
st.text('powered by Tensorflow 2.0 and Transformers 2.9.1')

st.info('Wait to loading model base')
model_t5 = TFT5ForConditionalGeneration.from_pretrained('t5-base')
task_specific_params = model_t5.config.task_specific_params
if task_specific_params is not None:
    model_t5.config.update(task_specific_params.get("summarization", {}))

model_t5.load_weights('model_base/saved_weight_model_large')
tokenizer = T5Tokenizer.from_pretrained('t5-base')
st.success("Successfully load model")

st.header('TEXT TO SUMMARIZE')
st.info('Press Ctrl + Enter to apply input text')
INPUT_TEXT = st.text_area("Text to summarize", key="input1")

if INPUT_TEXT != "":
    input_ids = tokenizer.encode(
        INPUT_TEXT, return_tensors="tf", max_length=512)
    line_input_text = INPUT_TEXT.splitlines()
    st.info("YOUR TEXT TO SUMMARIZE")

    st.write(INPUT_TEXT)
    st.header("YOUR TEXT SUMMARY")
    if st.checkbox('Extractive summarization', key='extractive'):
        st.info("Summarizing your text using extractive summarization. Please wait.....")
        start_time = time.time()
        extractive_summary = summarization(INPUT_TEXT)
        st.balloons()
        st.success("SUCCESSFULLY SUMMARIZE YOUR TEXT")
        elapse = time.time() - start_time
        st.info('Time to summary text in seconds: {:5.2f}'.format(elapse))
        st.success("HERE ARE THE SUMMARY TEXT")
        
        for i in extractive_summary:
            if i.text in INPUT_TEXT:
                st.markdown(i)
                st.markdown(f'***Line is IN the document***')
        

    if st.checkbox('Deep learning - Abtractive GENSUM summary', key='extractive'):
        st.info("Summarizing your text using abstractive summarization. Please wait.....")
        start_time = time.time()
        model_t5.config.max_length = 150
        pred = model_t5.generate(input_ids)
        summary = tokenizer.decode(
            pred[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        summary = summary.replace("<extra_id_0> ", "")
        summary = summary.replace(" . ", " .\n")
        st.balloons()
        st.success("SUCCESSFULLY SUMMARIZE YOUR TEXT")
        elapse = time.time() - start_time
        st.info('Time to summary text in seconds: {:5.2f}'.format(elapse))
        st.success("HERE ARE THE SUMMARY TEXT")
        
        lines = summary.splitlines()
        for line in lines:
            if line.endswith('.'):
                st.write(line)
                if line in INPUT_TEXT:
                    st.markdown('Line is in the document')
                else:
                    st.markdown('***Line is NOT IN the document***')


    # if st.checkbox('Comparison between Extractive summarization and Abstractive summarization', key='comparison'):
        
    #     col1, col2 = st.beta_columns(2)
    #     with col1:
    #         st.subheader('Extractive summarization')

    #     with col2:
    #         st.subheader('Abstractive summarization')
