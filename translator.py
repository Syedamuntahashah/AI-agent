
import streamlit as st
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig
from dotenv import load_dotenv
import os
import asyncio

# Load .env and GEMINI_API_KEY
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

# Async OpenAI Client for Gemini
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Define the model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

# Run configuration
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

# Utility function to handle asyncio in Streamlit
def run_agent_sync(agent, input_text, config):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(
        Runner.run(agent, input=input_text, run_config=config)
    )

# Streamlit UI setup
st.set_page_config(page_title="🌐 AI Translator", layout="centered")
st.markdown("""
    <h1 style='text-align: center;'>🌍 AI Translator Agent</h1>
    <p style='text-align: center; font-size: 18px;'>Translate English text to any language using Gemini + Langchain</p>
    <hr style='border-top: 2px solid #bbb;'/>
""", unsafe_allow_html=True)

# Supported target languages
languages = ["Urdu", "French", "Turkish", "Spanish", "Arabic", "German", "Chinese", "Hindi", "Russian", "Japanese"]

# Language selector
target_lang = st.selectbox("🌐 Choose Target Language", languages)

# Input field
input_text = st.text_area("✍️ Enter English Text", height=150)

# Translate button
if st.button("🔁 Translate"):
    if input_text.strip():
        with st.spinner("Translating..."):
            translator_agent = Agent(
                name="Translator Agent",
                instructions=f"You are a translator agent. Translate the input text from English to {target_lang}.",
            )
            response = run_agent_sync(translator_agent, input_text, config)
            st.success(f"✅ Translated Text in {target_lang}")
            st.text_area("📄 Output", value=response.final_output, height=150)
    else:
        st.warning("⚠️ Please enter text to translate.")
