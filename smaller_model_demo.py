import streamlit as st
import transformers
import time
import tensorflow as tf
from transformers import T5Tokenizer, TFT5Model, TFT5ForConditionalGeneration


st.title('ABSTRACTIVE TEXT SUMMARIZATION')
st.text('powered by Tensorflow 2.0 and Transformers 2.9.1')

st.info('Wait to loading model')
model_t5 = TFT5ForConditionalGeneration.from_pretrained('t5-small')
task_specific_params = model_t5.config.task_specific_params
if task_specific_params is not None:
    model_t5.config.update(task_specific_params.get("summarization", {}))

model_t5.load_weights('model/model_weights')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
st.success("Successfully load model")

st.header('TEXT TO SUMMARIZE')
INPUT_TEXT = st.text_area("Text to summarize", key="input1")

if INPUT_TEXT != "":
    input_ids = tokenizer.encode(
        INPUT_TEXT, return_tensors="tf", max_length=512)

    st.info("YOUR TEXT TO SUMMARIZE")

    st.write(INPUT_TEXT)

    st.info("Summarizing your text. Please wait.....")
    start_time = time.time()
    pred = model_t5.generate(input_ids)
    summary = tokenizer.decode(
        pred[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    summary = summary.replace("<extra_id_0> ", "")
    summary = summary.replace(" . ", " .\n")
    st.balloons()
    st.success("SUCCESSFULLY SUMMARIZE YOUR TEXT")
    elapse = time.time() - start_time
    st.info('Time to summary text: {:5.2f}'.format(elapse))
    st.success("HERE ARE THE SUMMARY TEXT")
    
    lines = summary.splitlines()
    for line in lines:
        if line.endswith('.'):
            st.write(line)
