import streamlit as st
import os
import pickle
from corpus_populator import load_corpus,update_corpus
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import time

client1=Groq(api_key=st.secrets['GROQ_API_KEY'])

@st.cache_resource
def load_cached_corpus():
    corpus=load_corpus()
    if corpus:
        corpus['model']=SentenceTransformer(corpus['model_name'])
        return corpus

st.set_page_config(page_title='NotesBot',page_icon='🤖',layout='wide',initial_sidebar_state="expanded")



if 'uploaded_files_list' not in st.session_state:
    st.session_state.uploaded_files_list=[]
data_fold='data'
os.makedirs(data_fold,exist_ok=True)

col1,col2=st.columns([3,1])
with col1:
    st.title('🤖 NotesBot',text_alignment='left')

with col2:
    mode=st.selectbox('Mode:',['Question','Revise'],key='mode_selector')
st.divider()
with st.sidebar:
    mmode = st.selectbox(
    "Choose model's mode",
    [
        "Default",
        "5 Year Old Mode",
        "Professional Mode",
        "Funny Mode",
        "Strict Teacher Mode",
        "Gen Z Mode"
    ],key='mmode_selector'
)



prompt_extender = ""

if mmode == "Default":
    prompt_extender = ""

elif mmode == "5 Year Old Mode":
    prompt_extender = "Explain everything like I'm 5 years old. Use very simple words and short sentences."

elif mmode == "Professional Mode":
    prompt_extender = "Respond in a formal, concise, and professional tone. Use technical accuracy where needed."

elif mmode == "Funny Mode":
    prompt_extender = "Add humor, light sarcasm, and make the response entertaining while staying correct."

elif mmode == "Strict Teacher Mode":
    prompt_extender = "Be strict, point out mistakes clearly, and explain corrections like a strict teacher."

elif mmode == "Gen Z Mode":
    prompt_extender = "Respond in a casual Gen Z style with modern slang, but keep it understandable."





if mode=='Question':
    with st.sidebar:

        no_of_retrievals=5
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
                    history+=f"{message['role']} {message['content']}\n"
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
                    prompt+='\n\n'+prompt_extender
                    for t in range(patience):
                        try:
                            response=client1.chat.completions.create(model='llama-3.3-70b-versatile',messages=[{'role':'user','content':prompt}])
                            answer=response.choices[0].message.content
                            st.session_state.messages.append({'role': 'assistant','content': answer})
                            with st.chat_message('assistant'):
                                st.markdown(answer)
                            break
                        except Exception as e:
                            if t==patience-1:
                                st.error(e)
                            else:
                                time.sleep(3)

                else:
                    st.write('No relevant chunks found :(')

else:
    if 'revise_history' not in st.session_state:
        st.session_state.revise_history=[]
    with st.sidebar:
        st.subheader("History")
        for i,past in enumerate(st.session_state.revise_history):
            if st.button(f"{past['mode']} - {', '.join(past['files'])}",use_container_width=True,key=f'history{i}'):
                st.session_state.selected_record=past
                st.rerun()


    if 'selected_record' in st.session_state:
        col1,col2=st.columns([9,1])
        with col1:
            st.write(st.session_state.selected_record['mode'])
            st.write(f"{', '.join(st.session_state.selected_record['files'])}")
        with col2:
            if st.button('<- Close',key='close_history'):
                del st.session_state.selected_record
                st.rerun()
        st.divider()
        st.markdown(st.session_state.selected_record['output'])
    else:
        corpus=load_corpus()
        if corpus is None:
            st.error("Embedd files first")
        else:
            selected_files=st.multiselect('Select the files for this session',corpus['files'],key='file_multiselect')
            if selected_files:
                st.divider()
            selected_chunks_text=[chunk['text'] for chunk in corpus['chunks'] if chunk['file'] in selected_files]

            content=" ".join(selected_chunks_text)
            estimated_tokens = len(content) // 4

            st.caption(f"Estimated tokens: {estimated_tokens:,}")
            
            presets = {
                "Summary": """
            Create a concise 200-word summary of the given content.
            Structure:
            - Lead with the main idea
            - Include 2-3 key supporting points
            - End with significance or conclusion
            Guidelines:
            - Use simple, direct language
            - Keep paragraphs short
            - No filler or unnecessary elaboration
            """,
                "Detailed Summary": """
            Create a detailed 500-word explanation of the given content.
            Structure:
            - Overview: What is this?
            - Main Concepts: Core ideas explained
            - Details: Supporting information and context
            - Practical Application: Why it matters
            Use bullet points, code blocks, or subheadings as needed for clarity.
            """,
                "Cheat Sheet": """
            Create a scannable cheat sheet for quick reference and revision.
            Include:
            - Key Concepts: Definitions and important ideas
            - Formulas/Rules: With brief explanations (use code blocks for syntax)
            - Examples: Practical examples with code if applicable
            - Common Pitfalls: Mistakes to avoid
            Use:
            - Bullet points for lists
            - Code blocks for syntax, commands, or technical notation
            - Short tables for comparisons
            - Bold for emphasis
            Keep it compact and revision-friendly.
            """,
                "MCQs": """
            Generate 10 multiple choice questions from the given content.
            Format for each question:
            Question: [Clear, exam-style question]\n
            A) [Option]
            B) [Option]
            C) [Option]
            D) [Option]\n
            Correct Answer: [A/B/C/D]
            Explanation: [Why correct, why others are wrong]
            Guidelines:
            - Include code snippets or syntax in questions if relevant
            - Use code blocks for technical examples
            - Test understanding, not just memorization
            - Make wrong answers plausible
            - Vary difficulty (easy to hard)
            """,
            
                "Study Guide": """
            Create a comprehensive study guide from the given content.
            Include:
            - Learning Objectives: What should be understood
            - Main Topics: Break content into logical sections with brief explanations
            - Examples: Code snippets, scenarios, or step-by-step walkthroughs
            - Summary Points: Key takeaways (bullet points)
            - Practice Problems: 3-5 problems or scenarios to apply knowledge
            Use code blocks, bullet points, and diagrams as appropriate.
            Make it self-contained for independent learning.
            """,
                "Quick Reference": """
            Create a quick reference card (one-page format).
            Use:
            - Short headings for organization
            - Code blocks for syntax and commands
            - Bullet points for lists
            - Tables for comparisons
            - Bold/highlight for critical info
            Keep everything terse but complete—just the essentials.
            """
            }            


            if 'revise_history' in st.session_state and st.session_state.revise_history:
                
                for item in st.session_state.revise_history:
                    with st.expander(f"{item['mode']} - {', '.join(item['files'])}"):
                        with st.chat_message('assistant'):
                            st.markdown(item['output'])
            


            def generate_study(generate_type,prompt,selected_files,content):
                history="\n".join([item['output'] for item in st.session_state.revise_history if item['mode']==generate_type])[-3000:]
                
                

                prompt=prompt+f'Notes:{content}\n history:{history} '
                prompt+='\n\n'+prompt_extender
                
                with st.spinner('Generating...'):
                    MAX_TOKENS = 80000
                    if estimated_tokens > MAX_TOKENS:
                        st.error(
                            f"Selected content is too large ({estimated_tokens:} estimated tokens).Please select fewer files.")
                        st.stop()
                    if not selected_files or not content:
                        st.error('No selected files  or content :(')
                        st.stop()
                    
                    patience=3
                    for t in range(patience):
                        try:
                            response=client1.chat.completions.create(model='meta-llama/llama-4-scout-17b-16e-instruct',messages=[{'role':'user','content':prompt}])    
                            answer=response.choices[0].message.content
                            st.session_state.revise_history.append({'mode':generate_type,'files': selected_files,'output':answer})
                            break
                        except Exception as e:
                            if t==patience-1:
                                st.error(e)
                            else:
                                time.sleep(3)
                    


            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📝 Summary", use_container_width=True, key="btn_summary"):
                    generate_study("Summary", presets["Summary"], selected_files,content)
            
            with col2:
                if st.button("📖 Detailed", use_container_width=True, key="btn_detailed"):
                    generate_study("Detailed Summary", presets["Detailed Summary"], selected_files,content)
            
            with col3:
                if st.button("📋 Cheat Sheet", use_container_width=True, key="btn_cheat"):
                    generate_study("Cheat Sheet", presets["Cheat Sheet"], selected_files,content)
            
            col4, col5, col6 = st.columns(3)
            
            with col4:
                if st.button("❓ MCQs", use_container_width=True, key="btn_mcqs"):
                    generate_study("MCQs", presets["MCQs"], selected_files,content)
            
            with col5:
                if st.button("🎴 Study Guide", use_container_width=True, key="btn_sg"):
                    generate_study("Flashcards", presets["Study Guide"], selected_files,content)
            
            with col6:
                if st.button("🎤 Quick Reference", use_container_width=True, key="btn_qf"):
                    generate_study("Quick Reference", presets["Quick Reference"], selected_files,content)
            
            st.divider()
            
            st.subheader("Custom Prompt")

            custom_prompt = st.text_area(
                "Ask anything about these notes:",
                placeholder="e.g., Explain the components of a CPU",
                height=80,
                label_visibility="collapsed",
                key="custom_study_prompt"
            )
            
            if st.button("Generate", type="primary", use_container_width=True, key="btn_custom_gen"):
                if not custom_prompt.strip():
                    st.error("Enter a prompt")
                else:
                    
                    generate_study("Custom", custom_prompt, selected_files,content)
            
            st.divider()
        # History section (at bottom)
        if st.session_state.revise_history:
            past = st.session_state.revise_history[-1]
            with st.chat_message('assistant'):
                st.markdown(past['output'])
            
            

            


        



        








