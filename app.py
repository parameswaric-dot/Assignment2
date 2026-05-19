import streamlit as st
from transformers import pipeline

# ---------------------------------
# Load Pretrained Model
# ---------------------------------
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-large"
)

# ---------------------------------
# Streamlit Page Config
# ---------------------------------
st.set_page_config(
    page_title="International Marketing Prompt App",
    page_icon="🌍",
    layout="centered"
)

# ---------------------------------
# Title
# ---------------------------------
st.title("🌍 International Marketing Prompt App")

st.markdown(
    "Generate AI-powered international business marketing content."
)

# ---------------------------------
# User Input
# ---------------------------------
product = st.text_input(
    "Enter Product Name",
    placeholder="Example: Smart Watch"
)

# ---------------------------------
# Generate Button
# ---------------------------------
if st.button("Generate Marketing Content"):

    if product.strip() == "":
        st.warning("Please enter a product name.")

    else:

        with st.spinner("Generating professional marketing content..."):

            # Strong Detailed Prompt
            prompt = f"""
You are a world-class international business marketing strategist.

Create premium marketing content for the following product.

Product Name: {product}

Generate detailed content using this exact format.

Global Product Title:
(Create a premium international product title)

Marketing Slogan:
(Create a catchy global slogan)

Emotional Marketing Description:
(Write 4-5 professional sentences emotionally connecting with customers)

Luxury Brand Description:
(Write 4-5 premium luxury branding sentences)

Social Media Advertisement:
(Write a modern viral advertisement with hashtags and emojis)

Make the content creative, realistic, persuasive, and business-professional.
"""

            # Generate Response
            result = generator(
                prompt,
                max_length=512,
                temperature=0.95,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2
            )

            output = result[0]["generated_text"]

            # Display
            st.subheader("Generated Marketing Content")

            st.success("Content Generated Successfully!")

            st.markdown(output)