import os
from frontend import config
import numpy as np
import pandas as pd
import streamlit as st
from tqdm import tqdm
from typing import List, Dict
import textwrap
from dataclasses import dataclass
from PIL import Image
import time
from ratelimit import limits, sleep_and_retry
import io
from dotenv import load_dotenv
import fitz
from google import genai
from google.genai import Client, types

@dataclass
class Config:
    MODEL_NAME: str = 'gemini-2.0-flash-exp'
    TEXT_EMBEDDING_MODEL_ID: str = 'text-embedding-004'
    DPI: int = 300

class PDFProcessor:
    @staticmethod
    def pdf_to_images(pdf_path: str, dpi: int) -> List[Image.Image]:
        pdf_document = fitz.open(pdf_path)
        images = []
        for page_number in range(pdf_document.page_count):
            page = pdf_document[page_number]
            pix = page.get_pixmap(matrix = fitz.Matrix(dpi / 72, dpi / 72))
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            images.append(image)
        pdf_document.close()
        return images
    
class GeminiClient:
    def __init__(self, api_key: str):
        self.client = Client(api_key=api_key)
    
    @property
    def models(self):
        return self

    def embed_content(self, model: str, contents: list[str], config: types.EmbedContentConfig):
        return self.client.models.embed_content(
            model=model, 
            contents=contents, 
            config=config
        )

    def generate_content(self, model: str, contents: list[str]):
        return self.client.models.generate_content(
            model=model, 
            contents=contents
        )
    
    def make_prompt(self, element: str) -> str:
        return f"""You are an agent tasked with summarizing research tables and texts
                from research papers for retrieval. These summaries will be embedded and used 
                to retrieve the raw text or table elements. Give a concise summary of the 
                tables or text that is well optimized for retrieval. 
                Table or text: {element}"""
    
    def analyze_page(self, image: Image.Image) -> str:
        prompt = """Analyze this document image and:
                1. Extract all visible text
                2. Describe any tables, their structure and content
                3. Explain any graphs or figures
                4. Note any important formatting or layout details
                Provide a clear, detailed description that captures all key information."""
        
        return self.generate_content(
            model=Config.MODEL_NAME,
            contents=[prompt, image]
        ).text
        
    def find_best_passage(self, query: str, df: pd.DataFrame) -> Dict:
        try:
            query_response = self.embed_content(
                model=Config.TEXT_EMBEDDING_MODEL_ID,
                contents=[query],
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )

            query_embedding = np.array(query_response.embeddings[0].values)

            similarities = [
                np.dot(query_embedding, np.array(page_embedding))
                for page_embedding in df['Embeddings']
            ]

            best_idx = np.argmax(similarities)

            if not df.iloc[best_idx]['Analysis']:
                raise ValueError("Selected passage is empty")
            
            return {
                'page': best_idx,
                'content': df.iloc[best_idx]['Analysis']
            }
        except Exception as e: 
            st.error(f"Error finding the best passage: {e}")
            return {
                'page': 0,
                'content': "Error: Could not find relevant passage"
            }
        
    def make_answer_prompt(self, query: str, passage: dict) -> str:
        escaped = passage['content'].replace("'", "").replace("\n", " ")
        return textwrap.dedent(f"""
            You are a helpful assistant analyzing research papers.
            Use the provided passage to answer the question.
            Be comprehensive but explain technical concepts clearly.
            If the passage is irrelevant, say so.

            QUESTION: '{query}'
            PASSAGE: '{escaped}'

            ANSWER:
        """)
        
@sleep_and_retry
@limits(calls=1, period=1)
def create_embeddings(self, data: str):
    time.sleep(1)
    return self.client.models.embed_content(
        model = Config.TEXT_EMBEDDING_MODEL_ID,
        contents = data,
        config = type.EmbedContentConfig(task_type = "RETRIEVAL_DOCUMENT")
    )

class RAGApplication:
    def __init__(self, api_key: str):
        self.gemini_client = GeminiClient(api_key=api_key)
        self.data_df = None
    
    def process_pdf(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")
        
        Images = PDFProcessor.pdf_to_images(pdf_path, Config.DPI)
        
        page_analyses = []
        st.write("Analyzing PDF pages...")
        for i, image in enumerate(tqdm(Images)):
            analysis = self._analyze_image(image)
            if analysis:
                page_analyses.append(analysis)
        
        if not page_analyses:
            raise ValueError("No content could be extracted from the PDF")
        
        self.data_df = pd.DataFrame({
            'Analysis': page_analyses
        })

        st.write("\n Generating embeddings")
        embeddings = []
        try:
            for text in tqdm(self.data_df['Analysis']):
                embed_result = self.gemini_client.embed_content(
                    model=Config.TEXT_EMBEDDING_MODEL_ID,
                    contents=[text],
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
                )
                if embed_result and embed_result.embeddings:
                    embeddings.append(embed_result.embeddings[0].values)
                else:
                    raise ValueError("Failed to generate embeddings")
                
            if len(embeddings) != len(self.data_df):
                raise ValueError(f"Generated {len(embeddings)} embeddings for {len(self.data_df)} texts")
            
            self.data_df['Embeddings'] = embeddings

        except Exception as e:
            print(f"Error generating embeddings: {e}")
            time.sleep(10)
    
    def _analyze_image(self, image: Image.Image) -> str:
        return self.gemini_client.analyze_page(image)
    
    def answer_questions(self, questions: List[str]) -> List[Dict[str,str]]:
        if self.data_df is None:
            raise ValueError("Please process a PDF first using process_pdf() method")
        
        answers = []
        for question in questions:
            try:
                passage = self.gemini_client.find_best_passage(question, self.data_df)
                prompt = self.gemini_client.make_answer_prompt(question, passage)
                response = self.gemini_client.generate_content(
                    model = Config.MODEL_NAME,
                    contents = [prompt]
                )
                answers.append({
                    'question': question,
                    'answer': response.text,
                    'source': f"Page {passage['page']}\nContent: {passage['content']}"
                })
            except Exception as e:
                st.error(f"Error processing question: {e}")
                answers.append({
                    'question': question,
                    'answer': f"Error generating answer: {str(e)}",
                    'source': "Error"
                })
        return answers
        
def main():
    load_dotenv()
    st.set_page_config(page_title="RAG AI Application", page_icon="📚", layout="wide")
    st.title("Ask the RAG AI Application")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY API key is required")
        return
    
    try:
        app = RAGApplication(api_key)
    except Exception as e:
        st.error(f"Failed to initialize application: {e}")
        return
    
    with st.form(key = "my_form"):
        pdf_file = st.file_uploader("Upload a PDF file", type = ['pdf'])
        questions = st.text_input("Enter your question:", placeholder = "Please provide a short summary")
        submit_button = st.form_submit_button(label = "Submit")
    
    if submit_button and pdf_file and questions:
        try:
            temp_pdf_path = f"temp_{pdf_file.name}"
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            
            st.write("Processing PDF...")
            with st.spinner("Analyzing document..."):
                app.process_pdf(temp_pdf_path)
                answers = app.answer_questions([questions])
            
            st.write("### Answers:")
            for answer in answers:
                st.write(f"**Question**: {answer['question']}")
                st.write(f"**Answer**: {answer['answer']}")
                st.write(f"*Source*: {answer['source']}")
        
        except Exception as e:
            st.error(f"Error processing the PDF: {e}")
        finally:
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

if __name__ == "__main__":
    main()