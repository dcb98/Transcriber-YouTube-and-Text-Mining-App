import os
import string
import requests
import pandas as pd
import streamlit as st
from pytube import YouTube
from zipfile import ZipFile
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

st.markdown('# 📝 **Transcriber App**')
bar = st.progress(0)


# global varible
with open("stopwords pandemic.txt", "r") as f:
     data = f.read()
     must_exist_stopwords = data.split("\n")
     f.close()

porter = PorterStemmer()
stopwords = set(STOPWORDS)
vectorizer = TfidfVectorizer()

# Custom functions 

# 2. Retrieving audio file from YouTube video
def get_yt(URL):
    video = YouTube(URL)
    yt = video.streams.get_audio_only()
    yt.download()

    #st.info('2. Audio file has been retrieved from YouTube video')
    bar.progress(10)

# 3. Upload YouTube audio file to AssemblyAI
def transcribe_yt():

    current_dir = os.getcwd()

    for file in os.listdir(current_dir)[::-1]:
        if file.endswith(".mp4"):
            mp4_file = os.path.join(current_dir, file)
            #print(mp4_file)
    filename = mp4_file
    bar.progress(20)

    def read_file(filename, chunk_size=5242880):
        with open(filename, 'rb') as _file:
            while True:
                data = _file.read(chunk_size)
                if not data:
                    break
                yield data
    headers = {'authorization': api_key}
    response = requests.post('https://api.assemblyai.com/v2/upload',
                            headers=headers,
                            data=read_file(filename))
    audio_url = response.json()['upload_url']
    #st.info('3. YouTube audio file has been uploaded to AssemblyAI')
    bar.progress(30)

    # 4. Transcribe uploaded audio file
    endpoint = "https://api.assemblyai.com/v2/transcript"

    json = {
    "audio_url": audio_url
    }

    headers = {
        "authorization": api_key,
        "content-type": "application/json"
    }

    transcript_input_response = requests.post(endpoint, json=json, headers=headers)

    #st.info('4. Transcribing uploaded file')
    bar.progress(40)

    # 5. Extract transcript ID
    transcript_id = transcript_input_response.json()["id"]
    #st.info('5. Extract transcript ID')
    bar.progress(50)

    # 6. Retrieve transcription results
    endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    headers = {
        "authorization": api_key,
    }
    transcript_output_response = requests.get(endpoint, headers=headers)
    #st.info('6. Retrieve transcription results')
    bar.progress(60)

    # Check if transcription is complete
    from time import sleep

    while transcript_output_response.json()['status'] != 'completed':
        sleep(5)
        st.warning('Transcription is processing ...')
        transcript_output_response = requests.get(endpoint, headers=headers)
    
    bar.progress(100)

    # 7. Print transcribed text
    st.header('Output')
    st.success(transcript_output_response.json()["text"])

    # 8. Save transcribed text to file

    # Save as TXT file
    yt_txt = open('yt.txt', 'w')
    yt_txt.write(transcript_output_response.json()["text"])
    yt_txt.close()

    # Save as SRT file
    srt_endpoint = endpoint + "/srt"
    srt_response = requests.get(srt_endpoint, headers=headers)
    with open("yt.srt", "w") as _file:
        _file.write(srt_response.text)
    
    zip_file = ZipFile('transcription.zip', 'w')
    zip_file.write('yt.txt')
    zip_file.write('yt.srt')
    zip_file.close()

def preprocess_text(text):
    
    # Casefolding
    casefolding = text.casefold()
    casefolding = casefolding.split()
    # casefolding = casefolding.split()
    # print(Casefolding)

    # Tokenization
    tokenization = " ".join(casefolding).split()
    # print(Tokenization)

    # Filtering
    ## Punctuation
    filter_punct = str.maketrans("", "", string.punctuation)
    punctuatization = [word.translate(filter_punct) for word in tokenization]
    # print(Filtering)

    # Stemming
    stemming = [porter.stem(word) for word in punctuatization]
    # print(Stemming)

    # stopwords filtering
    final_text = [i for i in stemming if i in must_exist_stopwords]

    result = {
        "casefold" : casefolding,
        "tokenize" : tokenization,
        "punctuation" : punctuatization,
        "stemming" : stemming
    }
    
    result = pd.DataFrame(result)

    return result, final_text

def generate_wordcloud(word_token):
    wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate(" ".join(word_token))
    fig = plt.figure(figsize = (10, 10))
    plt.imshow(wordcloud)
    plt.axis("off")

    return fig

def readme():
    description = st.empty()
    description.markdown("""
    # About Me :
    tulisan disini menggunakan format markdown,
    bisa dibaca referensi nya [disini](https://www.markdownguide.org/basic-syntax/)
    """)

#####

# The App

# 1. Read API from text file
api_key = st.secrets['api_key']

#st.info('1. API is read ...')
st.warning('Awaiting URL input in the sidebar.')


# Sidebar
st.sidebar.header('Input Mode')
selection_mode = st.sidebar.selectbox("",["About", "Trancriber App", "NLP App"])

if selection_mode == "About":
    readme()
    st.empty()

if selection_mode == "Trancriber App":
    st.sidebar.header('Input parameter')

    with st.sidebar.form(key='my_form'):
        URL = st.text_input('Enter URL of YouTube video:')
        submit_button = st.form_submit_button(label='Go')

    # Run custom functions if URL is entered 
    if submit_button:
        get_yt(URL)
        transcribe_yt()

        with open("transcription.zip", "rb") as zip_download:
            btn = st.download_button(
                label="Download ZIP",
                data=zip_download,
                file_name="transcription.zip",
                mime="application/zip"
            )

if selection_mode == "NLP App":
    st.sidebar.header('Input text')
    with st.sidebar.form(key='my_form'):
        URL = st.selectbox(
            'Enter URL of Description Text:', 
            [" "] + [i for i in os.listdir() if "txt" in i and i not in ["requirements.txt", "api.txt"]]
            )
        submit_button = st.form_submit_button(label='Go')

    # Run custom function if URL is available and entered
    if submit_button:
        with open(URL, 'r') as f:
            data = f.read()
            f.close() 

        # text preprocessing
        st.subheader("Text Preprocessing")
        data_text, cleaned_text = preprocess_text(data)
        st.dataframe(data_text)

        # text separation
        final_text = []
        iteration  = 0
        temp_list  = []

        for word in cleaned_text:
            if iteration > 10:
                final_text.append(" ".join(temp_list))
                iteration = 0
                temp_list = []
            elif iteration < 10:
                temp_list.append(word)
                iteration += 1
            else:
                final_text.append(" ".join(temp_list))
                iteration = 0
                temp_list = []

        # TF-IDF algorithm
        st.subheader("TF-IDF Weighting")
        result_weight = vectorizer.fit_transform(final_text)
        text_names, weights = vectorizer.get_feature_names(), result_weight.toarray()
        data_weights = pd.DataFrame.from_records(weights)
        data_weights = data_weights.set_axis(text_names, axis = 1, inplace = False)
        st.dataframe(data_weights)
        
        print(cleaned_text)

        # WordCloud
        st.subheader("Wordcloud Visualisation")
        wordcloud_visual = generate_wordcloud(cleaned_text)
        st.pyplot(wordcloud_visual)
