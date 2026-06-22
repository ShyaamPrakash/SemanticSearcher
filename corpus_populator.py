import pickle 
import os
from sentence_transformers import SentenceTransformer
import pdfplumber
import numpy as np
from pathlib import Path


def chunker(text, chunk_size=200, overlap=20):
    words = text.split()
    chunks = []
    overlap_words = int(chunk_size * overlap / 100)
    step_size = chunk_size - overlap_words
    
    for start_idx in range(0, len(words), step_size):
        end_idx = min(start_idx + chunk_size, len(words))
        chunk_txt = " ".join(words[start_idx:end_idx])

        
        chunks.append({'text': chunk_txt, 'length': len(chunk_txt.split())})
    
    return chunks

def extract_pdf(filepath):
    pages={}
    try:
        with pdfplumber.open(filepath) as pdf:
            for pagenum,page in enumerate(pdf.pages):
                pgtext=page.extract_text()
                if pgtext:
                    pages[pagenum]=pgtext
        return pages
    except Exception as e:
        print("Unable to read pdf due to ",e)

def extract_txt(filepath):
    try:
        with open(filepath,'r',encoding='utf-8') as f:
            pgtext=f.read()

    except UnicodeDecodeError:
        with open(filepath,'r',encoding='latin-1') as f:
            pgtext=f.read()

    except Exception as e:
        print("Unable to process txt file due to ",e)

    return {0:pgtext} if pgtext.split() else {}
    
def extract_and_chunk(filepath):
    if filepath.lower().endswith('.pdf'):
        pages=extract_pdf(filepath)

    elif filepath.lower().endswith('.txt'):
        pages=extract_txt(filepath)
    
    else:
        print("Wrong file type please choose correct files")
        return None
    all_chunks=[]
    filename=Path(filepath).name
    if not pages:
        return None
    for pageno,pgtext in pages.items():
        chunks=chunker(pgtext,chunk_size=50,overlap=20)
        for chunk in chunks:
            chunk['pageno']=pageno+1
            chunk['file']=filename

        all_chunks.extend(chunks)
    return all_chunks

def update_corpus(model_name="all-MiniLM-L6-v2",data_dir='data'):

    if not os.path.exists(data_dir):
        print("Data folder not found")
        return None
    
    files=[f for f in os.listdir(data_dir) if f.endswith(('.pdf','.txt'))]
    if not files:
        print("No files found")
        return None
    print(f'{len(files)} files found!')
    corpus=[]
    try:
        for file in files:
            filepath=os.path.join(data_dir,file)
            filechunks=extract_and_chunk(filepath)
            if not filechunks:
                print(f"No information found in the file {file}")
                continue
            else:
                corpus.extend(filechunks)
                print(f'{len(filechunks)} chunks added')

    except Exception as e:
        print('Error:',e)
    if not corpus:
        print("No chunks added")
        return None
    
    print("Loading model")
    model=SentenceTransformer(model_name)
    print("Embedding corpus")
    corpus_texts=[c['text'] for c in corpus]
    embeddings=model.encode(corpus_texts)
    print("embeddings shape:",embeddings.shape)

    corpus_data={'model_name':model_name,'embeddings':embeddings,'chunks':corpus,'files':files,'metadata':{'total_chunks':len(corpus),'total_files':len(files),'embedding_shape':embeddings.shape[1]}}

    os.makedirs('.',exist_ok=True)
    try:
        with open('corpus.pkl','wb') as f:
            pickle.dump(corpus_data,f)
        print("Saved successfully")
    except Exception as e:
        print("Error ",e)
        
    return corpus_data

def load_corpus():
    if not os.path.exists('corpus.pkl'):
        return None
    else:
        with open('corpus.pkl','rb') as f:
            corpus=pickle.load(f)
        print(f"Loaded - embeddings shape: {corpus['embeddings'].shape}")
        return corpus
    
if __name__=='__main__':
    corpus=update_corpus()



