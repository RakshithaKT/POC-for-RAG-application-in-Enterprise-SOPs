import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from groq import Groq
import base64
import os
from PIL import Image 
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(
    page_title="Multimodal Research RAG",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Multimodal Research Assistant")
st.write("Query your research papers visually using OpenCLIP embeddings and Llama 3.2 Vision via Groq.")


st.sidebar.header("Configuration")

groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    groq_client = Groq()
else:
    st.sidebar.warning("Please enter your Groq API key to enable text generation.")


@st.cache_resource
def init_chroma():
    client = chromadb.PersistentClient(path="./chroma_db")
    embedding_function = OpenCLIPEmbeddingFunction()
    image_loader = ImageLoader()
    
    collection = client.get_collection(
        name="research_papers_multimodal",
        embedding_function=embedding_function,
        data_loader=image_loader
    )
    return collection

try:
    collection = init_chroma()
    st.sidebar.success(f"Connected to ChromaDB! Current page count: {collection.count()}")
except Exception as e:
    st.sidebar.error(f"Failed to load ChromaDB. Ensure 'chroma_db' folder is in your root directory. Error: {e}")


def encode_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


user_query = st.text_input("Ask a question about your research papers (e.g., 'Show me the graph on revenue trends'):")

if user_query:
    if not groq_api_key:
        st.error("Please provide your Groq API key in the sidebar before executing a query.")
    else:
        with st.spinner("Searching the vector database..."):
            results = collection.query(
                query_texts=[user_query],
                n_results=3,
                include=["uris", "metadatas"]
            )
            
        if results and results['ids'][0]:
            # Extract the lists of all retrieved matches (up to 3)
            ids = results['ids'][0]
            uris = results['uris'][0]
            metadatas = results['metadatas'][0]
            
            # Layout Split 
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📍 Retrieved Context Pages")
                
                # Loop through and display all 3 retrieved images in the UI
                for i in range(len(ids)):
                    st.caption(f"**Match {i+1}:** {metadatas[i].get('source_paper')} (Page {metadatas[i].get('page_number')})")
                    filename = os.path.basename(uris[i])
                    local_uri = os.path.join(".", "paper_images", filename)
                    
                    try:
                        image = Image.open(local_uri)
                        st.image(image, use_container_width=True)
                    except Exception as img_err:
                        st.error(f"Could not render image: {img_err}")
            
            with col2:
                st.subheader("🤖 LLM Analysis")
                with st.spinner("Llama 4 Scout is analyzing all retrieved pages..."):
                    try:
                        # 1. Start the Groq prompt with your text instructions
                        prompt_content = [
                            {"type": "text", "text": f"Analyze these document images and answer the following question: {user_query}. Be concise and factually stick ONLY to what is visible in these pages."}
                        ]
                        
                        # 2. Loop through the retrieved images and append each one to the Groq prompt
                        for uri in uris:
                            filename = os.path.basename(uri)
                            local_uri = os.path.join(".", "paper_images", filename)
                            base64_img = encode_image_base64(local_uri)
                            
                            prompt_content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                            })
                        
                        # 3. Send the combined payload to Groq
                        response = groq_client.chat.completions.create(
                            model="meta-llama/llama-4-scout-17b-16e-instruct",
                            messages=[{"role": "user", "content": prompt_content}],
                            temperature=0.1
                        )
                        
                        st.markdown(response.choices[0].message.content)
                        
                    except Exception as api_err:
                        st.error(f"Groq API Error: {api_err}")
        else:
            st.warning("No relevant matching pages found in the database.")