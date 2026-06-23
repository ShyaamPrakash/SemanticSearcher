import streamlit as st
import os
import pickle
from corpus_populator import load_corpus,update_corpus
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import time

client=Groq(api_key=st.secrets['GROQ_API_KEY'])

@st.cache_resource
def load_cached_corpus():
    corpus=load_corpus()
    if corpus:
        corpus['model']=SentenceTransformer(corpus['model_name'])
        return corpus

st.set_page_config(page_title='NotesBot',page_icon='🤖',layout='wide')
st.title('🤖 NotesBot',text_alignment='left')
st.divider()

if 'uploaded_files_list' not in st.session_state:
    st.session_state.uploaded_files_list=[]
data_fold='data'
os.makedirs(data_fold,exist_ok=True)


with st.sidebar:
    no_of_retrievals=st.selectbox('No of chunks to retrive(Retrives more context)',[1,2,3,4,5])
    st.divider()
    try:
        uploaded_files=st.file_uploader('Upload the documents (pdf or txt)',type=['pdf','txt'],accept_multiple_files=True)
        if uploaded_files:
            st.write('Uploaded files')
            for file in uploaded_files:
                file_path=os.path.join(data_fold,file.name)
                if file.name not in st.session_state.uploaded_files_list:
                    with open(file_path,'wb') as f:
                        f.write(file.getvalue())
                    st.session_state.uploaded_files_list.append(file.name)
        for no,file in enumerate(st.session_state.uploaded_files_list,1):
                st.write(f'{no}.{file}')
    except Exception as e:
        print("Error:",e)
    st.divider()
    st.write("Files in data folder")
    dfiles=sorted([f for f in os.listdir(data_fold) if f.endswith(('.pdf','.txt'))])
    col1,col2=st.columns([3,1])
    if dfiles:
        for file in dfiles:
            with col1:
                st.write('📁',file)
            with col2:
                if st.button('🗑️',key=f'delete {file}'):
                    os.remove(os.path.join(data_fold,file))
                    if file in st.session_state.uploaded_files_list:
                        st.session_state.uploaded_files_list.remove(file)
                    st.success("File deleted successfully")
                    st.rerun()
    else:
        st.write("No files found :(")
    st.divider()
    if st.button("Embed all files",key='embed_button',use_container_width=True,type='primary'):
        with st.spinner("Your files are being embedded"):
            try:
                corpus=update_corpus()
                if corpus:
                    st.success("Files embedded successfully")
                    load_cached_corpus.clear()
                    st.rerun()
                else:
                    st.error("Failed to build corpus")
            except Exception as e:
                st.error(e)

    st.divider()
    if st.button("View analytics",key='analytics'):
        st.write('Analytics')
        corpus=load_corpus()
        if corpus:
            st.metric('Total files',corpus['metadata']['total_files'])
            st.metric('Total chunks',corpus['metadata']['total_chunks'])
            st.metric('Embeddings shape',corpus['metadata']['embedding_shape'])
            st.subheader('Files in corpus')
            for f in corpus['files']:
                st.write(f)
        else:   
            st.write("Corpus not found")

        
if 'messages' not in st.session_state:
        st.session_state.messages=[]



    
corpus=load_cached_corpus()

if corpus is None:
    st.warning("No corpus found. Upload and embed files first.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.write(message['content'])
    if query:=st.chat_input("Ask any question"):
        st.session_state.messages.append({'role':'user','content':query})
        with st.chat_message('user'):
            st.write(query)
        if corpus:
            embedded_query=corpus['model'].encode(query)
                
            similarities=cosine_similarity([embedded_query],corpus['embeddings'])[0]
            top_indices=similarities.argsort()[::-1][:no_of_retrievals]
            
               
            retrieved_chunks=[]
            for idx in top_indices:
                retrieved_chunks.append({
                    'chunk': corpus['chunks'][idx],
                    'confidence': similarities[idx]})
                
            history=''
            for message in st.session_state.messages[-6:]:
                history+=f'{message['role']} {message['content']}\n'
            if retrieved_chunks:
                context='\n\n'
                for item in retrieved_chunks:
                    oneanswer = f"\n• {item['chunk']['text']} \n (File:{item['chunk']['file']}) \n (Page no:{item['chunk']['pageno']})\n (Confidence:{item['confidence']:.2%})"
                    context+=oneanswer+'\n\n'
                if max(item['confidence'] for item in retrieved_chunks)<0.4:
                    prompt=f"""Answer the question using your general knowledge.History of conversation:{history}Question:{query}Note:The document search did not find sufficiently relevant information.Rules:
                                        - Do not reference any uploaded documents.
                                        - Do not invent citations.
                                        - Give a concise, accurate answer.Make it neatly formated"""
                else:
                    prompt=f"""Answer the question using the provided sources.History of conversation:{history}Question:{query}Sources:{context}Rules:
                                        - Use the sources as the primary evidence.
                                        - Include citations in the form:
                                        [File: filename, Page: page_number]
                                        - If multiple sources support the answer, cite multiple sources.
                                        - Do not invent citations.
                                        - Keep the answer clear and concise.It must be formated neatly
                                        -If all the citations are from teh same file then mention it once alone and mention page numbers alone.
                                        """
                patience=3
                for t in range(patience):
                    try:
                        response=client.chat.completions.create(model='llama-3.3-70b-versatile',messages=[{'role':'user','content':prompt}])
                        answer=response.choices[0].message.content
                        st.session_state.messages.append({'role': 'assistant','content': answer})
                        with st.chat_message('assistant'):
                            st.text(answer)
                        break
                    except:
                        if t==patience-1:
                            st.write('Groq servers are overloaded currently.Please try again at a later time')
                        else:
                            time.sleep(3)

            else:
                st.write('No relevant chunks found :(')




    








