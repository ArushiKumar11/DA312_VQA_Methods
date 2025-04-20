import os
import sys
import streamlit as st
from PIL import Image
import torch
import nltk
import io

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MultimodalVQAModel
from src.utils import init_nltk, setup_device, create_multimodal_preprocessors, process_single_example

# Set page configuration
st.set_page_config(
    page_title="Visual Question Answering",
    page_icon="🔍",
    layout="wide"
)

# Initialize NLTK resources
@st.cache_resource
def initialize_resources():
    init_nltk()
    return True

# Load model and resources
@st.cache_resource
def load_model_resources():
    # Set up device
    device = setup_device()
    
    # Define paths
    checkpoint_path = os.environ.get(
        "CHECKPOINT_PATH", 
        "../checkpoints/bert_vit/checkpoint-1560/model.safetensors"
    )
    answer_space_path = os.environ.get(
        "ANSWER_SPACE_PATH", 
        "../dataset/answer_space.txt"
    )
    
    # Check if the path exists, try alternative paths
    if not os.path.exists(checkpoint_path):
        alternatives = [
            "checkpoint-1560/model.safetensors",
            "checkpoints/checkpoint-1560/model.safetensors",
            "checkpoint/bert_vit/checkpoint-1560/model.safetensors",
        ]
        for alt in alternatives:
            if os.path.exists(alt):
                checkpoint_path = alt
                break
    
    if not os.path.exists(answer_space_path):
        alternatives = [
            "data/answer_space.txt",
            "answer_space.txt",
        ]
        for alt in alternatives:
            if os.path.exists(alt):
                answer_space_path = alt
                break
    
    # Load answer space
    with open(answer_space_path, "r") as f:
        answer_space = f.read().splitlines()
    
    # Create preprocessors
    text_model = "bert-base-uncased"
    image_model = "google/vit-base-patch16-224-in21k"
    tokenizer, preprocessor = create_multimodal_preprocessors(text_model, image_model)
    
    # Load model
    model = MultimodalVQAModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        num_labels=len(answer_space),
        map_location=device
    )
    model.to(device)
    model.eval()
    
    return model, tokenizer, preprocessor, answer_space, device

def main():
    # Initialize
    resources_initialized = initialize_resources()
    
    # Header
    st.title("Visual Question Answering Demo")
    st.markdown("""
    Upload an image and ask a question about it. The model will attempt to answer your question based on the visual content.
    """)
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info("""
    This app demonstrates a Visual Question Answering (VQA) model based on BERT and ViT Transformers.
    
    The model takes an image and a question as input and provides an answer based on understanding both the visual and textual content.
    """)
    
    # Load model resources
    with st.spinner("Loading model..."):
        try:
            model, tokenizer, preprocessor, answer_space, device = load_model_resources()
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.stop()
    
    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    # Display image and question input
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if uploaded_file is not None:
            # Save the uploaded file temporarily
            image_bytes = uploaded_file.read()
            image = Image.open(io.BytesIO(image_bytes))
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Save to a temporary file
            temp_image_path = "temp_image.png"
            image.save(temp_image_path)
    
    with col2:
        if uploaded_file is not None:
            st.subheader("Ask a question about the image")
            question = st.text_input("Question", 
                                    placeholder="e.g., How many chairs are there in the room?")
            
            if question:
                with st.spinner("Thinking..."):
                    # Process the example
                    result = process_single_example(
                        image_path=temp_image_path,
                        question=question,
                        tokenizer=tokenizer,
                        preprocessor=preprocessor,
                        model=model,
                        answer_space=answer_space,
                        device=device
                    )
                    
                    # Display the answer
                    st.subheader("Answer")
                    st.markdown(f"### {result['answer']}")
                    
                    # Display top predictions
                    st.subheader("Top Predictions")
                    for pred in result['top_predictions']:
                        st.markdown(f"{pred['rank']}. **{pred['answer']}** (Confidence: {pred['confidence']:.4f})")
    
    # Example images section
    st.markdown("---")
    
    
if __name__ == "__main__":
    main()