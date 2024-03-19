import streamlit as st
import pandas as pd
from streamlit_chat import message
from streamlit.components.v1 import html

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json 
 
@st.cache(allow_output_mutation=True)
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache(allow_output_mutation=True)
def get_dataset():
    df = pd.read_csv('wellness_dataset.csv')
    df['embedding'] = df['embedding'].apply(json.loads)
    return df


model = cached_model()
df = get_dataset()

st.header('Kevin 심리상담 챗봇')
st.markdown("❤️  좋은 질문과  답변, 정보를 주시면 Update 약속   호주에서  Kevin,  ")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('고민이 무엇인가요 ? ', '')
    submitted = st.form_submit_button('전송  /  Enter ')
    


if submitted and user_input:
    embedding = model.encode(user_input)

    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    st.session_state.past.append(user_input)
    st.session_state.generated.append(answer['챗봇'])

for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    if len(st.session_state['generated']) > i:
       message(st.session_state['generated'][i], key=str(i) + '_bot')
 
