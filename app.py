import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# ----------------------------------------
# Load Model and Tokenizer
# ----------------------------------------
model_name = "google/flan-t5-base"

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# ----------------------------------------
# Streamlit Config
# ----------------------------------------
st.set_page_config(
    page_title="International Marketing Prompt App",
    page_icon="🌍"
)

# ----------------------------------------
# Title
# ----------------------------------------
st.title("🌍 International Marketing Prompt App")

st.write("Generate professional international marketing content using AI.")

# ----------------------------------------
# Product Input
# ----------------------------------------
product = st.text_input(
    "Enter Product Name",
    placeholder="Example: Smart Watch"
)

# ----------------------------------------
# Generate Content
# ----------------------------------------
if st.button("Generate Marketing Content"):

    if product.strip() == "":
        st.warning("Please enter a product name.")

    else:

        with st.spinner("Generating content..."):

            # Strong Prompt
            prompt = f"""
You are a professional international marketing strategist.

Generate detailed business marketing content for the product: {product}

Include:

1. Global Product Title
2. Marketing Slogan
3. Emotional Marketing Description
4. Luxury Brand Description
5. Social Media Advertisement

Write professionally with creative business language.
"""

            # Tokenize Input
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )

            # Generate Response
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.9,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2
            )

            # Decode Output
            response = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            # Display Output
            st.subheader("Generated Marketing Content")

            st.success("Content Generated Successfully!")

            st.write(response)