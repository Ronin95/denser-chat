from openai import OpenAI
import streamlit as st
import time
import os
import subprocess
import argparse
import json
import logging
from urllib.parse import urlencode, quote
from dotenv import load_dotenv

# Import our helper functions from upload_pdf.py
from upload_pdf import ensure_upload_folder, append_file_to_sources

# Import modules for the retriever
from denser_retriever.keyword import (
    ElasticKeywordSearch,
    create_elasticsearch_client,
)
from denser_retriever.retriever import DenserRetriever
from google import genai

logger = logging.getLogger(__name__)
load_dotenv()

# Define available models
MODEL_OPTIONS = {
    "GPT-4": "gpt-4o",
    "Gemini 2.0 Flash": "gemini-2.0-flash"
}
context_window = 128000

# Get API keys from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
if not (openai_api_key or google_api_key):
    raise ValueError("Neither OPENAI_API_KEY nor GOOGLE_API_KEY environment variables is set")

openai_client = OpenAI(api_key=openai_api_key)
google_client = genai.Client(api_key=google_api_key)
history_turns = 5

prompt_default = (
    "### Instructions:\n"
    "You are a professional AI assistant. The following context consists of an ordered list of sources. "
    "If you can find answers from the context, use the context to provide a response in German. "
    "You must cite passages in square brackets [X] where X is the passage number (do not include passage word, only digit numbers)."
    "If you cannot find the answer from the sources, use your knowledge to come up with a reasonable answer in German. "
    "If the query asks to summarize the file or uploaded file, provide a summarization in German based on the provided sources. "
    "If the conversation involves casual talk or greetings, rely on your knowledge for an appropriate response in German. "
)

def sanitize_filename(filename):
    """
    Replaces spaces and the characters ü, ä, ö (and their uppercase variants)
    with underscores.
    """
    for ch in [" ", "ü", "ä", "ö", "Ü", "Ä", "Ö"]:
        filename = filename.replace(ch, "_")
    return filename

def create_viewer_url_by_passage(passage):
    """Create a URL to open PDF.js viewer with annotation highlighting."""
    base_url = "http://localhost:8000/viewer.html"
    try:
        ann_list = json.loads(passage[0].metadata.get('annotations', '[]'))
        pdf_url = passage[0].metadata.get('source', None)
        if not pdf_url or not ann_list:
            return None
        viewer_annotations = []
        for ann in ann_list:
            viewer_annotations.append({
                'x': ann.get('x', 0),
                'y': ann.get('y', 0),
                'width': ann.get('width', 0),
                'height': ann.get('height', 0),
                'page': ann.get('page', 0)
            })
        params = {
            'file': pdf_url,
            'annotations': json.dumps(viewer_annotations),
            'pageNumber': viewer_annotations[0]['page'] + 1
        }
        return f"{base_url}?{urlencode(params, quote_via=quote)}"
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error in create_viewer_url_by_passage: {e}")
        return None

def post_process_html(full_response: str, passages: list) -> str:
    """Replace citations with clickable HTML links."""
    import re
    def replace_citation(match):
        num = int(match.group(1)) - 1
        if num < len(passages):
            source = passages[num][0].metadata.get('source', '')
            if source.startswith(('http://', 'https://')) and not source.endswith('.pdf'):
                return f'<a href="{source}" target="_blank">[{num + 1}]</a>'
            else:
                viewer_url = create_viewer_url_by_passage(passages[num])
                if viewer_url:
                    return f'<a href="{viewer_url}" target="_blank">[{num + 1}]</a>'
                else:
                    return f'[{num + 1}]'
        return match.group(0)
    processed_text = re.sub(r'\[(\d+)\]', replace_citation, full_response)
    return processed_text

def stream_response(selected_model, messages, passages):
    """
    Stream assistant response based on the selected model.
    """
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        if selected_model == "gpt-4o":
            print("Using OpenAI GPT-4 model")
            messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})
            for response in openai_client.chat.completions.create(
                    model=selected_model,
                    messages=messages,
                    stream=True,
                    top_p=0,
                    temperature=0.0
            ):
                full_response += response.choices[0].delta.content or ""
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        else:
            print("Using Google Gemini model")
            response = google_client.models.generate_content(
                model=selected_model,
                contents=messages[-1]['content']
            )
            full_response = response.text
            message_placeholder.markdown(full_response, unsafe_allow_html=True)
    print(f"### messages\n: {messages}\n")
    print(f"### full_response\n: {full_response}\n")
    processed_response = post_process_html(full_response, passages)
    print(f"### processed_response\n: {processed_response}\n")
    st.session_state.messages.append({"role": "assistant", "content": processed_response})
    st.session_state.passages = passages
    st.rerun()

def main(args):
    st.set_page_config(layout="wide")
    
    # Initialize session state to keep track of already processed file names
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []

    # Sidebar: Title, description, and display list of uploaded PDFs from sources.txt
    st.sidebar.title("Navigation")
    st.sidebar.markdown("You can chat with all the uploaded files here.")
    sources_file = "sources.txt"
    if os.path.exists(sources_file):
        st.sidebar.subheader("Uploaded PDFs:")
        with open(sources_file, "r", encoding="utf-8") as f:
            for line in f:
                file_path = line.strip()
                if file_path:
                    file_name = os.path.basename(file_path)
                    st.sidebar.markdown(f"- **{file_name}**  \n`{file_path}`")
    
    # PDF uploader block: Save file, sanitize filename, append path, and trigger build.py
    uploaded_pdf = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_pdf is not None:
        safe_filename = sanitize_filename(uploaded_pdf.name)
        # Only process if this file hasn't been handled already.
        if safe_filename not in st.session_state.uploaded_files:
            st.session_state.uploaded_files.append(safe_filename)
            upload_folder = ensure_upload_folder("uploadedPDF")
            file_path = os.path.join(upload_folder, safe_filename)
            with open(file_path, "wb") as f:
                f.write(uploaded_pdf.getvalue())
            st.sidebar.write("Uploaded file saved as:", file_path)
            relative_path = os.path.join(".", upload_folder, safe_filename).replace("\\", "/")
            append_file_to_sources(relative_path)
            
            # Run build.py to process sources.txt and update the index.
            try:
                import sys
                result = subprocess.run(
                    [sys.executable, "build.py", "sources.txt", "output", "test_index"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                st.sidebar.write("Index built successfully:")
                st.sidebar.code(result.stdout)
            except subprocess.CalledProcessError as e:
                st.sidebar.write("Error building index:")
                st.sidebar.code(e.stderr)
    
    global retriever
    retriever = DenserRetriever(
        index_name=args.index_name,
        keyword_search=ElasticKeywordSearch(
            top_k=100,
            es_connection=create_elasticsearch_client(
                url="http://localhost:9200",
                username="elastic",
                password="",
            ),
            drop_old=False,
            analysis="default"
        ),
        vector_db=None,
        reranker=None,
        embeddings=None,
        gradient_boost=None,
        search_fields=["annotations:keyword"],
    )
    
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "Gemini 2.0 Flash"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "passages" not in st.session_state:
        st.session_state.passages = []
    
    # Main content: Chat demo interface
    st.title("FFG Chat Demo")
    selected_model_name = st.selectbox(
        "Select Model",
        options=list(MODEL_OPTIONS.keys()),
        key="model_selector",
        index=list(MODEL_OPTIONS.keys()).index(st.session_state.selected_model)
    )
    st.session_state.selected_model = selected_model_name
    # st.caption(
    #     "Try question \"What is example domain?\", \"What is in-batch negative sampling?\" or \"what parts have stop pins?\""
    # )
    st.divider()
    
    for i in range(len(st.session_state.messages)):
        with st.chat_message(st.session_state.messages[i]["role"]):
            st.markdown(st.session_state.messages[i]["content"], unsafe_allow_html=True)
    
    query = st.chat_input("Please input your question")
    if query:
        with st.chat_message("user"):
            st.markdown(query)
        start_time = time.time()
        passages = retriever.retrieve(query, 5, {})
        retrieve_time_sec = time.time() - start_time
        st.write(f"Retrieve time: {retrieve_time_sec:.3f} sec.")
        prompt = prompt_default + f"### Query:\n{query}\n"
        if len(passages) > 0:
            prompt += "\n### Context:\n"
            for i, passage in enumerate(passages):
                prompt += f"#### Passage {i + 1}:\n{passage[0].page_content}\n"
        context_limit = 4 * context_window if args.language == "en" else context_window
        prompt = prompt[:context_limit] + "### Response:"
        messages = st.session_state.messages[-history_turns * 2:]
        messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "user", "content": query})
        stream_response(MODEL_OPTIONS[selected_model_name], messages, passages)

def parse_args():
    parser = argparse.ArgumentParser(description='Denser Chat Demo')
    parser.add_argument('--index_name', type=str, default=None,
                        help='Name of the Elasticsearch index to use')
    parser.add_argument('--language', type=str, default='en',
                        help='Language setting for context window (en or ch, default: en)')
    parser.add_argument('--static_dir', type=str, default='static',
                        help='Directory where PDF.js and PDFs are served from')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
