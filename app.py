import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to get response from Gemini Pro
def get_gemini_response(input_text):
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(input_text)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Function to extract text from PDF
def input_pdf_text(uploaded_file):
    try:
        reader = pdf.PdfReader(uploaded_file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
        return text
    except Exception as e:
        return f"Error extracting PDF text: {str(e)}"

# Prompt Template
input_prompt = """
Hey, act like a skilled Application Tracking System (ATS) with deep expertise in tech fields, including software engineering, data science, data analysis, and big data engineering. Your task is to evaluate the resume based on the provided job description. Consider a highly competitive job market and provide actionable insights to improve the resume. Assign a percentage match based on the job description and identify missing keywords with high accuracy.

resume: {text}
description: {jd}

Return the response as a JSON string with the following structure:
{{"JD Match":"%","MissingKeywords":[],"Profile Summary":""}}
"""

# Streamlit App
st.set_page_config(page_title="Smart ATS", page_icon="📄", layout="wide")
st.title("Smart ATS - Resume Evaluator")
st.markdown("Optimize your resume for ATS and increase your chances in a competitive job market!")

# Input fields
st.subheader("Job Description")
jd = st.text_area("Paste the Job Description here", height=200, placeholder="Enter the job description...")

st.subheader("Upload Resume")
uploaded_file = st.file_uploader("Upload Your Resume (PDF only)", type="pdf", help="Please upload a PDF file of your resume.")

# Submit button
submit = st.button("Evaluate Resume", type="primary")

# Handle submission
if submit:
    if uploaded_file is not None and jd:
        with st.spinner("Analyzing resume..."):
            # Extract text from PDF
            resume_text = input_pdf_text(uploaded_file)
            if resume_text.startswith("Error"):
                st.error(resume_text)
            else:
                # Format the prompt with resume text and job description
                formatted_prompt = input_prompt.format(text=resume_text, jd=jd)
                
                # Get response from Gemini
                response = get_gemini_response(formatted_prompt)
                
                if response.startswith("Error"):
                    st.error(response)
                else:
                    try:
                        # Parse the JSON response
                        response_json = json.loads(response)
                        
                        # Display results
                        st.subheader("Resume Analysis Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("JD Match", f"{response_json.get('JD Match', 'N/A')}")
                        
                        with col2:
                            st.write("**Missing Keywords**")
                            missing_keywords = response_json.get('MissingKeywords', [])
                            if missing_keywords:
                                st.write(", ".join(missing_keywords))
                            else:
                                st.write("None")
                        
                        st.write("**Profile Summary**")
                        st.write(response_json.get('Profile Summary', 'No summary provided.'))
                        
                    except json.JSONDecodeError:
                        st.error("Error parsing response from model. Please try again.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {str(e)}")
    else:
        st.warning("Please upload a resume and provide a job description before submitting.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Google Gemini | © 2025 Smart ATS")
