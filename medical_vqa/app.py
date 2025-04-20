import os
import torch
import numpy as np
import streamlit as st
from PIL import Image
from transformers import CLIPImageProcessor, GPT2Tokenizer
from model.model import VQAModel
from model.utils import generate_beam

# Set page configuration
st.set_page_config(
    page_title="Medical VQA",
    page_icon="🏥",
    layout="wide"
)

@st.cache_resource
def load_model(model_path):
    """Load the VQA model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQAModel().to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    model.load_state_dict(checkpoint)
    model.eval()
    
    return model, device

@st.cache_resource
def load_processors():
    """Load the image processor and tokenizer"""
    image_processor = CLIPImageProcessor.from_pretrained('flaviagiammarino/pubmed-clip-vit-base-patch32')
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return image_processor, tokenizer

def preprocess_image(image, image_processor):
    """Preprocess the image for the model"""
    return image_processor(image, return_tensors="pt")

def preprocess_question(question, tokenizer):
    """Preprocess the question for the model"""
    m = [
        torch.tensor(tokenizer.encode('question: ')),
        torch.tensor(tokenizer.encode(' context:')),
        torch.tensor(tokenizer.encode('answer '))
    ]
    
    m_mask = [
        torch.ones(len(tokenizer.encode('question: '))),
        torch.ones(len(tokenizer.encode(' context:'))),
        torch.ones(len(tokenizer.encode('answer ')))
    ]
    
    q = torch.tensor(tokenizer.encode(str(question)))
    q, q_mask, _ = make_padding_test_setting(24, q)
    q_len = m[0].size(0) + q.size(0) + m[1].size(0)
    q = torch.cat((m[0], q, m[1], torch.ones(1), m[2]))
    q_mask = torch.cat((m_mask[0], q_mask, m_mask[1], torch.ones(1), m_mask[2]))
    
    return q, q_mask, q_len

def make_padding_test_setting(max_len, tokens, do_padding=False):
    """Make padding for test setting"""
    padding = max_len - tokens.size(0)
    padding_len = 0
    
    if padding > 0:
        if do_padding:
            mask = torch.cat((torch.ones(tokens.size(0)), torch.zeros(padding)))
            tokens = torch.cat((tokens, torch.zeros(padding)))
            padding_len = padding
        else:
            mask = torch.ones(tokens.size(0))
    elif padding == 0:
        mask = torch.ones(max_len)
    elif padding < 0:
        tokens = tokens[:max_len]
        mask = torch.ones(max_len)
        
    return tokens, mask, padding_len

def predict(model, device, image, question, image_processor, tokenizer):
    """Generate answer for the given image and question"""
    # Preprocess image
    image_tensor = preprocess_image(image, image_processor)['pixel_values'].squeeze(0).to(device)
    
    # Preprocess question
    tokens, mask, q_len = preprocess_question(question, tokenizer)
    tokens = tokens.unsqueeze(0).long().to(device)
    mask = mask.unsqueeze(0).long().to(device)
    q_len = torch.tensor([q_len]).to(device)
    
    # Generate answer
    with torch.no_grad():
        embed = model.generate(image_tensor.unsqueeze(0), tokens, mask, q_len[0])
        out_text = generate_beam(model, tokenizer, generated=embed, entry_length=30, temperature=1)[0]
    
    # Clean the output text
    if "<|endoftext|>" in out_text:
        out_text = out_text.split("<|endoftext|>")[0]
    
    return out_text

def main():
    st.title("Medical VQA: Visual Question Answering for Medical Images")
    st.markdown("Upload a medical image and ask a question about it.")
    
    # Sidebar for model selection
    st.sidebar.title("Model Configuration")
    model_path = st.sidebar.text_input(
        "Model Path", 
        value="checkpoints/best_model.pth", 
        help="Path to the trained model checkpoint"
    )
    
    # Load model and processors
    try:
        model, device = load_model(model_path)
        image_processor, tokenizer = load_processors()
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        st.stop()
    
    # Upload image
    uploaded_file = st.file_uploader("Choose a medical image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Question input
        question = st.text_input("Ask a question about the image:", "")
        
        if st.button("Get Answer"):
            if question:
                with st.spinner("Generating answer..."):
                    answer = predict(model, device, image, question, image_processor, tokenizer)
                
                # Display the answer
                st.success("Answer Generated!")
                st.markdown(f"**Question:** {question}")
                st.markdown(f"**Answer:** {answer}")
            else:
                st.warning("Please enter a question.")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Example Questions")
    st.sidebar.markdown("- Is this an axial plane?")
    st.sidebar.markdown("- Is there evidence of any abnormalities?")
    st.sidebar.markdown("- What is the location of the lesion?")
    st.sidebar.markdown("- Is this a normal image?")

if __name__ == "__main__":
    main()