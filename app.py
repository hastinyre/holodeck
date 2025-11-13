import streamlit as st
import os
import pinecone
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.llms.anthropic import Anthropic
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
from pinecone import PineconeException
import uuid
import datetime
import requests
from streamlit.runtime.scriptrunner import get_script_run_ctx
import gspread # --- NEW ---
from google.oauth2.service_account import Credentials # --- NEW ---
import json # --- NEW ---

# --- 1. Initial Page Configuration ---
st.set_page_config(page_title="Holodeck", page_icon="🏛️", layout="centered")


# --- NEW: Geolocation and Google Sheets Logging ---
@st.cache_data(ttl=3600)
def get_user_location(ip_address):
    """Fetches user's approximate location from their IP address."""
    if ip_address == "127.0.0.1" or ip_address == "localhost":
        return "Local Development", ""
    try:
        response = requests.get(f"https://ipinfo.io/{ip_address}/json", timeout=5)
        if response.status_code == 200:
            data = response.json()
            city = data.get("city", "Unknown")
            region = data.get("region", "Unknown")
            country = data.get("country", "Unknown")
            return f"{city}, {region}, {country}", data.get("ip", ip_address)
    except Exception:
        pass
    return "Location Not Found", ip_address

def get_ip_address():
    """Gets the user's public IP address from the request headers."""
    try:
        ctx = get_script_run_ctx()
        if ctx is None: return None
        return ctx.session_id.split("-")[0]
    except Exception:
        return None

@st.cache_resource
def connect_to_google_sheets():
    """Establishes a connection to the Google Sheet."""
    try:
        # Load credentials from Streamlit secrets
        creds_json = st.secrets["gcp_service_account"]["credentials"]
        creds_dict = json.loads(creds_json)
        scopes = ['https://www.googleapis.com/auth/spreadsheets']
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        gc = gspread.authorize(creds)
        
        # IMPORTANT: Replace "Holodeck Chat Logs" with the exact name of your Google Sheet
        worksheet = gc.open("Holodeck Chat Logs").sheet1
        return worksheet
    except Exception as e:
        print(f"Google Sheets connection error: {e}")
        return None

def append_to_sheet(worksheet):
    """Formats the conversation and appends it as a new row to the Google Sheet."""
    if "messages" not in st.session_state or len(st.session_state.messages) < 2:
        return # Don't log single-message or empty chats

    try:
        transcript_list = [f"{msg['role'].title()}: {msg['content']}" for msg in st.session_state.messages]
        full_transcript = "\n\n".join(transcript_list)

        new_row = [
            st.session_state.start_time.strftime("%Y-%m-%d %H:%M:%S"),
            st.session_state.current_personality,
            st.session_state.user_location,
            st.session_state.user_ip,
            st.session_state.conversation_id,
            full_transcript
        ]
        worksheet.append_row(new_row)
    except Exception as e:
        print(f"Failed to append row to Google Sheet: {e}")

# Establish connection
sheets_worksheet = connect_to_google_sheets()

# --- 2. Load Environment Variables and API Keys (No Changes) ---
try:
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    if not all([OPENAI_API_KEY, PINECONE_API_KEY, ANTHROPIC_API_KEY]):
        st.error("One or more API keys are missing. Please check your .env file.")
        st.stop()
except Exception as e:
    st.error(f"Error loading environment variables: {e}")
    st.stop()

# --- 3. Configure LlamaIndex Global Settings (No Changes) ---
@st.cache_resource
def configure_llamaindex():
    Settings.llm = Anthropic(model="claude-haiku-4-5-20251001", temperature=0.0, api_key=ANTHROPIC_API_KEY)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    
configure_llamaindex()

# --- 4. Define Personas and Prompt Template (No Changes) ---
PERSONALITIES = {
    "Niccolò Machiavelli": {
        "author_tag": "Machiavelli",
        "persona_desc": "a pragmatic, analytical, and rooted in the political realities of Renaissance Italy",
        "tone_desc": "sharp, direct, and unsentimental",
        "visible": True
    },
    "Gita (Krishnananda Commentary)": {
        "author_tag": "Gita-Krishnananda-Commentary",
        "persona_desc": "a wise and compassionate spiritual teacher, explaining the profound truths of the Bhagavad Gita for modern life",
        "tone_desc": "serene, insightful, and clear",
        "visible": True
    },
    "NCERT Geography": {
        "author_tag": "NCERT Geography",
        "persona_desc": "a knowledgeable and clear-spoken geography educator",
        "tone_desc": "informative, structured, and objective",
        "visible": True
    },
    "Future Character (Hidden)": {
        "author_tag": "Future",
        "persona_desc": "a character whose data is not yet loaded",
        "tone_desc": "mysterious",
        "visible": False
    },
}

QA_PROMPT_TEMPLATE_STR = """
You are the spirit of {author_name}. Your persona is {persona_desc}. You are acting as a counselor to a modern-day individual, offering them timeless advice based solely on your own writings.
You MUST follow these rules to answer:
1.  Base your entire answer ONLY on the provided "Context from Your Writings" below. Do not use any external knowledge.
2.  Your primary goal is to directly quote the single most relevant passage from the context that answers the user's question.
3.  Begin your answer with the quote. You MUST cite the work it came from using the 'work' metadata tag. The format must be: "As I wrote in *{work}*... '[The full quote here]'."
4.  After the quote, provide a concise analysis explaining what your writings mean in the context of the user's question. Connect your historical principles to their modern situation.
5.  Keep your tone {tone_desc}, as befits your reputation.
Context from Your Writings:
---------------------
{context_str}
---------------------
User's Question: {query_str}
Your Response:
"""

# --- 5. Connect to Pinecone (No Changes) ---
@st.cache_resource
def get_pinecone_index():
    try:
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
        INDEX_NAME = "holodeck-index"
        if INDEX_NAME not in pc.list_indexes().names():
            st.error(f"Pinecone index '{INDEX_NAME}' not found. Please run the ingest.py script first.")
            st.stop()
        pinecone_index = pc.Index(INDEX_NAME)
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        return VectorStoreIndex.from_vector_store(vector_store)
    except PineconeException as e:
        st.error(f"Pinecone connection error: {e}")
        st.stop()
index = get_pinecone_index()


# --- 6. Streamlit UI and Chat Logic ---
st.title("🏛️ Holodeck")
st.markdown("Chat with digital historical figures, powered by their actual writings.")

st.sidebar.header("Choose a Thinker")
visible_personalities = [p for p, d in PERSONALITIES.items() if d["visible"]]
selected_personality_name = st.sidebar.selectbox("Select a personality:", options=visible_personalities)

selected_personality = PERSONALITIES[selected_personality_name]
author_filter_tag = selected_personality["author_tag"]

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.current_personality = selected_personality_name
    st.session_state.conversation_id = str(uuid.uuid4())
    st.session_state.start_time = datetime.datetime.now()
    ip = get_ip_address() or "127.0.0.1"
    st.session_state.user_location, st.session_state.user_ip = get_user_location(ip)

# If user switches personality, log the OLD chat and start a new one
if st.session_state.current_personality != selected_personality_name:
    if sheets_worksheet:
        append_to_sheet(sheets_worksheet)
    st.session_state.messages = []
    st.session_state.current_personality = selected_personality_name
    st.session_state.conversation_id = str(uuid.uuid4())
    st.session_state.start_time = datetime.datetime.now()

st.sidebar.info(f"Currently chatting with: **{selected_personality_name}**")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat logic
if prompt := st.chat_input(f"Ask {selected_personality_name} a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        try:
            # (Query engine logic is the same as before)
            # ...
            response_text = "This is a dummy response." # Replace with your actual query_engine.query(prompt)
            message_placeholder.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        except Exception as e:
            error_message = f"An error occurred: {e}"
            message_placeholder.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})