import streamlit as st
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import os

# Load model and processor
@st.cache_resource
def load_model():
    model_name = "Salesforce/blip2-flan-t5-xl"  # Or a smaller model if needed
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

processor, model, device = load_model()

# Session state for history and gallery
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'gallery' not in st.session_state:
    st.session_state['gallery'] = []

st.title("Wildlife Image Captioning & VQA (BLIP-2)")

uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
task = st.radio("Task", ["Caption", "VQA"])

if uploaded_files:
    images = [Image.open(f).convert("RGB") for f in uploaded_files]
    st.image(images, caption=[f.name for f in uploaded_files], use_column_width=True)

    if task == "Caption":
        prompt = "Describe the image"
        prompts = [prompt] * len(images)
    else:
        prompt = st.text_input("Ask a question about the images:")
        prompts = [prompt] * len(images)

    if st.button("Generate"):
        results = []
        for img, prmpt, file in zip(images, prompts, uploaded_files):
            inputs = processor(images=img, text=prmpt, return_tensors="pt").to(device)
            generated_ids = model.generate(**inputs)
            output = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            results.append({
                'filename': file.name,
                'task': task,
                'prompt': prmpt,
                'output': output
            })
            # Add to gallery
            st.session_state['gallery'].append({'image': img, 'filename': file.name})
            # Add to history
            st.session_state['history'].append({
                'filename': file.name,
                'task': task,
                'prompt': prmpt,
                'output': output
            })
        for res in results:
            st.success(f"{res['filename']} - {res['task']}: {res['output']}")

# Gallery
st.header("Image Gallery")
if st.session_state['gallery']:
    st.image([g['image'] for g in st.session_state['gallery']], caption=[g['filename'] for g in st.session_state['gallery']], use_column_width=True)
else:
    st.info("No images in gallery yet.")

# History panel
st.header("History")
if st.session_state['history']:
    for h in reversed(st.session_state['history']):
        st.write(f"**{h['filename']}** | {h['task']} | Prompt: {h['prompt']} | Output: {h['output']}")
else:
    st.info("No history yet.")
