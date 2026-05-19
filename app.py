import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# -----------------------------
# Load Model
# -----------------------------
model_name = "google/flan-t5-base"

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="International Marketing Prompt App",
    page_icon="🌍",
    layout="centered"
)

# -----------------------------
# App Title
# -----------------------------
st.title("🌍 International Marketing Prompt App")

st.markdown("Generate AI-powered global marketing content instantly.")

# -----------------------------
# User Input
# -----------------------------
product = st.text_input("Enter Product Name")

# -----------------------------
# Generate Button
# -----------------------------
if st.button("Generate Marketing Content"):

    if product.strip() == "":
        st.warning("Please enter a product name.")
    else:

        with st.spinner("Generating marketing content..."):

            # Improved Prompt
            prompt = f"""
You are an expert international marketing strategist.

Create professional marketing content for the product: {product}

Include the following sections clearly:

1. Global Product Title
2. Catchy Marketing Slogan
3. Emotional Marketing Description
4. Luxury Brand Description
5. Social Media Advertisement

Write professionally and creatively.
"""

            # Tokenize
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=250,
                num_beams=5,
                early_stopping=True,
                temperature=0.9,
                do_sample=True
            )

            # Decode
            response = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            # Display
            st.subheader("Generated Marketing Content")

            st.success("Content Generated Successfully!")

            st.write(response)