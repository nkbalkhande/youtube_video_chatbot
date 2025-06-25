import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.retrievers import MultiQueryRetriever
from langchain.memory import ConversationBufferMemory
from langdetect import detect
from dotenv import load_dotenv
import os

load_dotenv()
gemini_key = os.getenv('Gemini_key')

st.set_page_config(page_title="YouTube Chatbot", layout="centered")
st.title("Chat with a YouTube Video")

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# User enters YouTube video ID
video_id = st.text_input("Enter YouTube Video ID:")


@st.cache_data(show_spinner=True)
def get_transcript_and_lang(video_id):
    transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
    transcript_obj = None

    for t in transcripts:
        if not t.is_generated:
            transcript_obj = t
            break

    if not transcript_obj:
        transcript_obj = transcripts.find_transcript(
            [t.language_code for t in transcripts])

    transcript_data = transcript_obj.fetch()
    transcript_text = " ".join([t.text for t in transcript_data])
    lang = detect(transcript_text)
    return transcript_text, lang


if video_id:
    try:
        transcript_text, lang = get_transcript_and_lang(video_id)
        st.info(f"Transcript Language Detected: {lang.upper()}")

        # Setup LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=gemini_key,
            temperature=0.4,
        )

        # Translate if not English
        if lang != "en":
            st.warning("Translating transcript to English...")
            translated_chunks = []
            for chunk in transcript_text.split(". "):
                prompt = f"Translate this to English:\n\n{chunk}"
                translated = llm.invoke(prompt).content
                translated_chunks.append(translated)
            transcript_text = " ".join(translated_chunks)

        # Create documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript_text])

        # Vector store
        embedding = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=gemini_key)
        vector_store = FAISS.from_documents(chunks, embedding)

        # Multi-query retriever
        multi_query = MultiQueryRetriever.from_llm(
            retriever=vector_store.as_retriever(search_kwargs={'k': 4}),
            llm=llm
        )

        # Prompt template
        prompt = PromptTemplate(
            template="""
                You are a helpful assistant.

                Chat History:
                {chat_history}

                ONLY use the transcript context below to answer the user's question.
                If the context is insufficient, say you don't know.

                Transcript:
                {context}

                 Question: {question}
                """,
            input_variables=["context", "question", "chat_history"]
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        def get_chat_history(memory):
            return "\n".join([f"{m.type.capitalize()}: {m.content}" for m in memory.chat_memory.messages])

        memory = st.session_state.memory  # local variable for safety
        chain = (
            RunnableParallel({
                "context": multi_query | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
                "chat_history": RunnableLambda(lambda _: get_chat_history(memory)),
            }) | prompt | llm | StrOutputParser()
        )

        # Chat interface
        user_input = st.chat_input("Ask a question about the video...")
        if user_input:
            with st.spinner("Generating answer..."):
                answer = chain.invoke(user_input)
                memory.chat_memory.add_user_message(user_input)
                memory.chat_memory.add_ai_message(answer)

                # ✅ Always show full chat history
            for msg in memory.chat_memory.messages:
                role = "user" if msg.type == "human" else "bot"
                with st.chat_message(role):
                    st.markdown(msg.content)

    except TranscriptsDisabled:
        st.error("This video has no transcripts.")
    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")
