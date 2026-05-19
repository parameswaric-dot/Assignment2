import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration

# -----------------------------------
# Load Model
# -----------------------------------
model_name = "google/flan-t5-base"

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# -----------------------------------
# Function to Generate Text
# -----------------------------------
def generate_text(prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.9,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

# -----------------------------------
# Streamlit Config
# -----------------------------------
st.set_page_config(
    page_title="International Marketing Prompt App",
    page_icon="🌍"
)

# -----------------------------------
# Title
# -----------------------------------
st.title("🌍 International Marketing Prompt App")

st.write("Generate professional international marketing content using AI.")

# -----------------------------------
# Input
# -----------------------------------
product = st.text_input(
    "Enter Product Name",
    placeholder="Example: Smart Watch"
)

# -----------------------------------
# Generate Button
# -----------------------------------
if st.button("Generate Marketing Content"):

    if product.strip() == "":
        st.warning("Please enter a product name.")

    else:

        with st.spinner("Generating marketing content..."):

            # Generate Each Section Separately

            title = generate_text(
                f"Create a premium global product title for {product}"
            )

            slogan = generate_text(
                f"Create a catchy marketing slogan for {product}"
            )

            emotional = generate_text(
                f"Write an emotional marketing advertisement for {product}"
            )

            luxury = generate_text(
                f"Write a luxury brand advertisement for {product}"
            )

            social = generate_text(
                f"Write a social media advertisement for {product} with hashtags"
            )

            # Display Output
            st.subheader("Generated Marketing Content")

            st.success("Content Generated Successfully!")

            st.markdown(f"## 🌟 Global Product Title")
            st.write(title)

            st.markdown(f"## 📢 Marketing Slogan")
            st.write(slogan)

            st.markdown(f"## ❤️ Emotional Marketing Description")
            st.write(emotional)

            st.markdown(f"## 💎 Luxury Brand Description")
            st.write(luxury)

            st.markdown(f"## 📱 Social Media Advertisement")
            st.write(social)