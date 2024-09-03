import os
import streamlit as st
import openai
import json
import re
from spellchecker import SpellChecker
from fuzzywuzzy import fuzz, process
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import config
from deep_translator import GoogleTranslator  

# Set page configuration at the very beginning
st.set_page_config(page_title="💬 CrescendoChat")

# Initialize the spell checker
spell = SpellChecker()

# Set the OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

# Display a welcome message
st.write("Welcome to 💬 CrescendoChat!")

# Function to load the dataset from a JSON file
def load_dataset_from_json(uploaded_file):
    try:
        data = json.load(uploaded_file)
        st.write("JSON file loaded successfully.")
        return data
    except json.JSONDecodeError as e:
        st.error(f"Failed to load JSON file: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# Function to clean and normalize text for matching
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

# Function to correct spelling in user input
def correct_spelling(text):
    corrected_text = []  # Corrected the assignment
    for word in text.split():
        corrected_word = spell.correction(word)
        if corrected_word is None:
            corrected_text.append(word)
        else:
            corrected_text.append(corrected_word)
    return ' '.join(corrected_text)

# Updated function to find the best matching prompt using fuzzy matching
def find_best_match(user_input, data):
    cleaned_input = clean_text(user_input)
    for item in data:
        for prompt in item['prompts']:
            cleaned_prompt = clean_text(prompt)
            score = fuzz.partial_ratio(cleaned_input, cleaned_prompt)
            if score > 70:
                return prompt, score, item['completion']

    return None, 0, "I'm sorry, I couldn't find an answer to that question."

# Function to translate text if necessary and infer the language
def translate_and_detect_language(text, target_lang='en'):
    translated_text = GoogleTranslator(source='auto', target=target_lang).translate(text)
    if translated_text.lower().strip() == text.lower().strip():
        return target_lang, text
    else:
        return 'fr', translated_text

# Function to clean up the response text
def clean_up_response(response_text):
    # Specific fix for known bad output
    response_text = re.sub(r'(\d+)\s*,\s*(\d{1,3})(\d{3})', r'\1,\2,\3', response_text)

    # Correcting issues with misplaced commas and spaces
    response_text = re.sub(r'(\d)\s+(\d)', r'\1\2', response_text)
    response_text = re.sub(r'(\d{1,3})(\d{3})(\d{3})', r'\1,\2,\3', response_text)

    # Further generic cleaning
    response_text = re.sub(r'\s+', ' ', response_text)
    response_text = response_text.replace(' ,', ',')
    response_text = response_text.replace(' .', '.')

    return response_text

# Sidebar for uploading the JSON file
with st.sidebar:
    st.title('💬 CrescendoChat')
    st.write("Upload your JSON file for fine-tuning:")
    uploaded_file = st.file_uploader("Choose a file...", type="json")

    if uploaded_file is not None:
        data = load_dataset_from_json(uploaded_file)
        if data:
            st.write("Data loaded and ready to be used.")
            st.session_state.prompt_responses = data
            st.session_state.model_initialized = True

# Set up the LangChain components
if st.session_state.get("model_initialized"):
    # Adjust the prompt to guide better output
    prompt_template = PromptTemplate(template="Please provide the provincial tax rates for Quebec in 2023 in a clear and precise format. Use the following format: 'The provincial tax rates in Quebec for 2023 are 15% on the first $45,105, 20% on the next $45,105, and so on.'")
    
    # Initialize ChatOpenAI with the API key and model explicitly specified
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",  # Ensure the model is specified
        openai_api_key=config.OPENAI_API_KEY  # Pass the API key explicitly
    )

    chat_chain = LLMChain(llm=llm, prompt=prompt_template)

    # Handle the question-answer interaction
    user_input = st.text_input("Ask me anything:")

    if user_input:
        detected_lang, translated_input = translate_and_detect_language(user_input)
        corrected_input = correct_spelling(translated_input)

        # Try to find a match in the custom data
        response_text = "Je suis désolé, je n'ai pas pu trouver de réponse à cette question."
        best_match, score, response_text = find_best_match(corrected_input, st.session_state.prompt_responses)
        if score <= 70:
            response_text = chat_chain.run(user_input=corrected_input)

        # Clean up the response text before displaying
        response_text = clean_up_response(response_text)

        # Translate the response back to French if the original input was in French
        if detected_lang != 'en':
            response_text = GoogleTranslator(source='en', target=detected_lang).translate(response_text)

        # Display the response with an icon and consistent style
        assistant_icon = "🧑‍💼"  # Corrected the assignment
        st.markdown(f"{assistant_icon} {response_text}")

else:
    st.write("Please upload a JSON file to initialize the chatbot.")
