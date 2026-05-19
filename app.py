import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model and tokenizer
model_name = "google/flan-t5-base"

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Streamlit UI
st.title("🌍 International Marketing Prompt App")

product = st.text_input("Enter Product Name")

if st.button("Generate Marketing Content"):

    prompt = f"""
    Product Name: {product}

    Generate:
    1. Global Product Title
    2. Marketing Slogan
    3. Emotional Marketing Description
    4. Luxury Branding Description
    5. Social Media Advertising Description
    """

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=200,
        temperature=0.8,
        do_sample=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.subheader("Generated Marketing Content")
    st.write(response)