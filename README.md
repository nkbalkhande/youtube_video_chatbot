#  YouTube Video Chatbot using LangChain, Gemini API, and Streamlit

This project allows users to **chat with any YouTube video** using its transcript. It uses **Google Gemini (Generative AI)** for translation and question-answering, and **LangChain** for managing the chatbot logic and retrieval. The UI is built using **Streamlit**.

---

##  Features

-  Extracts transcripts from any YouTube video using `youtube-transcript-api`
-  Detects transcript language and auto-translates to English if needed
-  Embeds transcript using Gemini Embeddings and stores in FAISS vector store
-  Uses LangChain with multi-query retriever for contextual question answering
-  Maintains chat history using LangChain Memory
-  Simple and interactive UI using Streamlit

---

## How It Works
- 1] User provides video ID

- 2] Transcript is fetched and language is detected

- 3] If not English, it's translated using Gemini

- 4] Transcript is split into chunks and embedded using Gemini Embeddings

- 5] FAISS vector store enables fast semantic search

- 6] MultiQueryRetriever fetches relevant chunks

- 7] LangChain assembles prompt using chat history + context

- 8] Gemini model answers the user's query

- 9] Chat history is displayed with Streamlit

## API Key Setup

To use this project, you need an API key from Google Gemini Large Language Model. Follow these steps to obtain and set up your API key:

1. **Get API Key:**
   - Visit Alkali App [Click Here](https://makersuite.google.com/app/apikey).
   - Follow the instructions to create an account and obtain your API key.

2. **Set Up API Key:**
   - Create a file named `.env` in the project root.
   - Add your API key to the `.env` file:
     ```dotenv
     GOOGLE_API_KEY=your_api_key_here
     ```

   **Note:** Keep your API key confidential. Do not share it publicly or expose it in your code.<br>

# ScreenShot :





##  Requirements

Install dependencies using `pip`:
```
streamlit
langchain
langchain-core
langchain-community
langchain-google-genai
google-api-python-client
youtube-transcript-api
faiss-cpu
python-dotenv
langdetect
```
## How to Run

- streamlit run youtube_chatbot.py
