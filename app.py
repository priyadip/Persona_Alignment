import streamlit as st
import requests
import json

st.set_page_config(
    page_title="Sherlock Holmes AI Detective",
    page_icon="🕵️",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* 1. FORCE DARK THEME & RESET */
    .stApp {
        background-color: #0e0e0e;
        color: #e0e0e0;
    }
    
    /* Global text setting to prevent dark-on-dark */
    p, h1, h2, h3, li {
        color: #e0e0e0;
    }

    /* 2. HEADER STYLING */
    .main-header {
        background: linear-gradient(180deg, #1a1a1a 0%, #0e0e0e 100%);
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #333;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .main-title {
        font-family: 'Georgia', serif;
        font-size: 3rem;
        font-weight: 700;
        color: #d4af37 !important; /* Gold */
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .subtitle {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 0.9rem;
        color: #888 !important;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    /* 3. CHAT BUBBLES - FIXING VISIBILITY */
    
    /* Remove default streamlit bubble styling */
    .stChatMessage {
        background-color: transparent !important;
        border: none !important;
    }

    /* USER MESSAGE (Right Side) */
    /* Target the container to reverse direction */
    [data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatar"] > div:contains("user")) {
        flex-direction: row-reverse;
    }
    
    /* Style the content box for USER */
    [data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatar"] > div:contains("user")) div[data-testid="stChatMessageContent"] {
        background-color: #3b82f6 !important; /* Bright Blue for contrast */
        border: 1px solid #2563eb;
        color: #ffffff !important; /* Force White Text */
        border-radius: 15px 15px 2px 15px;
    }
    
    /* Style the content box for SHERLOCK */
    [data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatar"] p:contains("🕵️‍♂️")) div[data-testid="stChatMessageContent"] {
        background-color: #1e1e1e !important; /* Dark Grey */
        border: 1px solid #444;
        border-left: 4px solid #d4af37 !important; /* Gold accent */
        color: #e0e0e0 !important; /* Light Grey Text */
        border-radius: 15px 15px 15px 2px;
    }
    
    /* FORCE TEXT COLORS INSIDE BUBBLES */
    /* This overrides Streamlit's default paragraph styling */
    div[data-testid="stChatMessageContent"] p {
        color: inherit !important;
    }

    /* 4. INPUT BAR FIX (Remove white bottom bar) */
    [data-testid="stBottom"] {
        background-color: #0e0e0e; /* Match main background */
        border-top: 1px solid #333;
    }
    
    .stChatInput textarea {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        border: 1px solid #444 !important;
    }
    
    /* 5. SIDEBAR POLISH */
    [data-testid="stSidebar"] {
        background-color: #121212;
        border-right: 1px solid #333;
    }
    
    /* Custom info box for sidebar to replace st.info */
    .sidebar-box {
        background-color: #1e1e1e;
        border-left: 3px solid #d4af37;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 class="main-title">SHERLOCK HOLMES</h1>
    <p class="subtitle">Consulting Detective • 221B Baker Street</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### About the Detective")

    st.markdown("""
    <div class="sidebar-box">
        <p style="margin:0; font-style: italic;">
        "I am Sherlock Holmes. I observe, I deduce, and I solve. State your case clearly."
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Methodology")
    st.markdown("""
    * **Observation:** Nothing escapes my eye.
    * **Deduction:** Eliminating the impossible.
    * **Science:** Forensic analysis.
    """)
    
    st.markdown("---")

    if st.button("🗑️ Clear Case History", type="primary"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.caption("Model: Qwen 2.5 7B (Fine-tuned) • Deployed on Modal")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "The game is afoot. What mystery brings you to my door today?"
    })

for message in st.session_state.messages:

    if message["role"] == "assistant":
        avatar = "🕵️‍♂️" 
    else:
        avatar = "user" 
        
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Describe the evidence..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🕵️‍♂️"):

        with st.spinner("🔍 Examining the evidence..."):
            try:

                API_URL = "https://m25csa023--sherlock-detective-sherlockmodel-generate-web.modal.run"

                if "YOUR_MODAL_URL_HERE" in API_URL:
                    st.error("⚠️ **System Error:** Modal URL not configured in code.")
                    st.stop()

                response = requests.post(
                    API_URL,
                    json={"prompt": prompt},
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    bot_response = data.get("response", "I cannot make a deduction from this data.")

                    st.markdown(bot_response)

                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": bot_response
                    })
                else:
                    st.error(f"My connection to the archives is severed. (Error {response.status_code})")
                    
            except Exception as e:
                st.error(f"Network Error: {str(e)}")
