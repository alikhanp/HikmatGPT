import streamlit as st
import pickle
import numpy as np
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI(
  api_key = st.secrets["API_KEY"]
)

@st.cache_resource
def load_model():
  model = SentenceTransformer('msmarco-distilbert-base-v4') #Stated to later get vector of question
  return model

@st.cache_resource
def load_excel_data():
  df = pd.read_excel('all_paragraphs_v1_final.xlsx') #Reads excel to get text to output
  paragraph = df.iloc[:, 0]
  book = df.iloc[:, 1]
  return df, paragraph, book

@st.cache_resource
def load_vector_data():
  infile = open('all_paragraphs_vector_v1.pkl', 'rb') #Gets text's vector to calcualte with question's vector
  embeddings_distilbert = pickle.load(infile)
  return infile, embeddings_distilbert

model = load_model()
df, paragraph, book = load_excel_data()
infile, embeddings_distilbert = load_vector_data()

def find_similar(vector_representation, all_representations, k=1): #Calculation function
    similarity_matrix = cosine_similarity(vector_representation, all_representations)
    similarities = similarity_matrix[0]
    if k == 1:
        return [np.argmax(similarities)]
    elif k is not None:
        return np.flip(similarities.argsort()[-k:][::1])

def run(): #Function that runs when button pressed. 
  search_vect = model.encode([search_string]) #Gets vector of question
  
  distilbert_similar_indexes = find_similar(search_vect, embeddings_distilbert, 10) #Finds the top 10 most similar vector correls
  rel = cosine_similarity(search_vect, embeddings_distilbert[0:]) #Calculates relevances to output
  
  output_data = [] #Declared arrays to enter text and book name data
  book_data = []
  
  for index in distilbert_similar_indexes: #Loop that appends the top 10 related text and book name vectors
    output_data.append(paragraph[index])
    book_data.append(book[index])
  
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=[
      {"role": "user", "content": "Read the following passages: " + str(output_data[:10])},
      {"role": "user", "content": "Write an answer to the question " + '"' + search_string + '"' + " from using the passages above"}
    ]
  )
  output = completion.choices[0].message.replace("\n", "\n")
  st.info(output) #Display ChatGPT's response
    
  st.subheader("References") #Display all 10 text references with book name and relevance
  for i in range(10):
    st.markdown(f":red[{output_data[i]}]")
    st.write(f"**Reference:** {book_data[i]}. **Relevance:** {str(round(rel[0][distilbert_similar_indexes[i]]*100, 2))}%")

st.title('HikmatGPT') #Set title, input, and caption for Steamlit
form = st.form("my_form")
search_string = form.text_area("What's your question?")
submitted = form.form_submit_button("Ask")
if submitted:
  st.write(run())
