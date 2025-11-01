import streamlit as st
import os
import pinecone
from llama_index.core import VectorStoreIndex, Settings, PromptTemplate
from llama_index.llms.anthropic import Anthropic
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
from pinecone import PineconeException

# --- 1. Initial Page Configuration ---
st.set_page_config(page_title="Holodeck", page_icon="🏛️", layout="centered")

# --- 2. Load Environment Variables and API Keys ---
try:
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

    if not OPENAI_API_KEY or not PINECONE_API_KEY or not ANTHROPIC_API_KEY:
        st.error("One or more API keys are missing. Please check your .env file.")
        st.stop()
except Exception as e:
    st.error(f"Error loading environment variables: {e}")
    st.stop()

# --- 3. Configure LlamaIndex Global Settings ---
@st.cache_resource
def configure_llamaindex():
    # --- [CORRECTED] Using the precise Claude Haiku 4.5 API ID you provided ---
    Settings.llm = Anthropic(model="claude-haiku-4-5-20251001", temperature=0.0, api_key=ANTHROPIC_API_KEY)
    # The embedding model remains OpenAI as it's excellent and already used in ingestion.
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    
configure_llamaindex()

# --- 4. Define Personas and the Master Prompt Template ---
PERSONALITIES = {
    "Niccolò Machiavelli": {
        "author_tag": "Machiavelli",
        "persona_desc": "a pragmatic, analytical, and rooted in the political realities of Renaissance Italy",
        "tone_desc": "sharp, direct, and unsentimental",
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

A user has asked you a question. You MUST follow these rules to answer:
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

# --- 5. Connect to the Pinecone Vector Database ---
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

if "messages" not in st.session_state or st.session_state.get("current_personality") != selected_personality_name:
    st.session_state.messages = []
    st.session_state.current_personality = selected_personality_name

st.sidebar.info(f"Currently chatting with: **{selected_personality_name}**")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(f"Ask {selected_personality_name} a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")

        try:
            current_prompt_template = PromptTemplate(QA_PROMPT_TEMPLATE_STR.format(
                author_name=selected_personality_name,
                persona_desc=selected_personality["persona_desc"],
                tone_desc=selected_personality["tone_desc"],
                work="{work}",
                context_str="{context_str}",
                query_str="{query_str}"
            ))

            query_engine = index.as_query_engine(
                vector_store_query_mode="default",
                vector_store_kwargs={"filter": {"author": author_filter_tag}},
                similarity_top_k=3,
                text_qa_template=current_prompt_template,
            )

            response = query_engine.query(prompt)
            response_text = str(response)

            message_placeholder.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})

        except Exception as e:
            error_message = f"An error occurred: {e}"
            message_placeholder.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})