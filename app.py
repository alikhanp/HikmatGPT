import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from revChatGPT.V1 import Chatbot

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
  
  prompt = " Read and take in these passages, only using their contents connect to the question: " + '"' + search_string + '". ' + str(output_data[:10])
  resp = ""
  for data in chatbot.ask(
    prompt
  ):
    resp = data["message"] #ChatGPT's response to prompt

  st.info(resp) #Display ChatGPT's response
  if (resp == ""):
    st.caption("**There is an issue with your access token. Try entering it again.**")
  elif ((resp.count(" ") > 1) and (resp.count(" ") < 50)) or (resp[-1] == "?"):
    st.caption("**If the question is not answered, try rewording it or changing/adding subjects.**")
    
  
  st.subheader("References") #Display all 10 text references with book name and relevance
  for i in range(10):
    st.markdown(f":red[{output_data[i]}]")
    st.write(f"**Reference:** {book_data[i]}. **Relevance:** {str(round(rel[0][distilbert_similar_indexes[i]]*100, 2))}%")

with st.sidebar:
  st.header("Login")
  access_token = st.text_input("**Enter your ChatGPT Access Token:**")
  st.caption("[Click here to find your ChatGPT Access Token](https://chat.openai.com/api/auth/session)")
  st.caption("Close sidebar when done")
if access_token:
  access_token = access_token.replace('"','')
  access_token = access_token.replace("'","")
  access_token = access_token.replace(" ","")
  chatbot = Chatbot(config={ #Login to ChatGPT
    "access_token": access_token
  })
  st.title('HikmatGPT') #Set title, input, and caption for Steamlit
  search_string = st.text_input("What's your question?")
  st.caption("Press Enter first to process text")
  if search_string: #Statement to check first if user pressed enter in textbox to process the question before the button shows
    ask = st.empty()
    if ask.button('Ask'):
      ask.empty()
      st.write(run())
else:
  st.header("Starting Instructions")
  st.subheader("Step 1: Log in to ChatGPT")
  st.markdown("Make sure you are Logged in to ChatGPT on [chat.openai.com/](https://chat.openai.com/chat)")
  st.image('step1.png')
  st.subheader("Step 2: Copy your Access Token")
  st.markdown("[Click here to find your ChatGPT Access Token](https://chat.openai.com/api/auth/session)")
  st.image('step2.PNG')
  st.markdown("Your Access token is in the quotations after 'accessToken':")
  st.subheader("Step 3: Paste your Access Token")
  st.image('step3.PNG')
  st.markdown("Paste it into the login sidebar and press enter")
