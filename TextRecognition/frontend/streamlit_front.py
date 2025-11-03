import io
import os
import base64

import httpx

from PIL import Image

import streamlit as st



#environmental variables
DB_API = os.getenv("DB_API")
INFERENCE_API = os.getenv("INFERENCE_API")

# Streamlit App Config
st.set_page_config(
    page_title="Text Recognition app",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Sidebar with instructions
st.sidebar.title("Instructions")
st.sidebar.info(
    """
    1. Upload an image (JPG, PNG, JPEG).  
    2. Click 'Predict' to get the model's generated text.  
    3. The prediction will appear below.
    """
)

# Main title
st.title("🖼️ Text recognition")
st.markdown(
    "Upload an image and get the text from Text Recognition model!"
)


# File Upload Section
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["png", "jpg", "jpeg"],
    help="Supported formats: PNG, JPG, JPEG"
)

if uploaded_file is not None:
    # Display uploaded image in a clean container
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Convert image to bytes for API
    image_bytes = io.BytesIO()
    image.save(image_bytes, format=image.format)
    image_bytes = image_bytes.getvalue()
    #convert it to base64 -> text
    image_64 = base64.b64encode(image_bytes).decode("utf-8")


    # Prediction Section
    if st.button("Predict"):
        with st.spinner("Sending image to model and getting prediction..."):
            try:
                #backend prediction API endpoint
                prediction_api_url = INFERENCE_API+'/prediction'
                #backend interaction API endpoint
                interaction_api_url = DB_API+'/interaction'

                # POST request with the image file
                response = httpx.post(
                    prediction_api_url,
                    json = {"image_64_base": image_64},
                    timeout = 10 
                )

                if response.status_code == 200:
                    prediction = response.json()
                    #push data to database
                    httpx.post(interaction_api_url, json = {"input": image_64, "prediction": prediction['generated_text']})
                    # Display generated text
                    st.success("✅ Prediction received!")
                    st.subheader("Generated text:")
                    st.write(f"**{prediction['generated_text']}**")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")

            except Exception as e:
                st.error(f"Request failed: {e}")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:gray;font-size:12px;'>Luchian's project 🖤</p>",
    unsafe_allow_html=True
)
