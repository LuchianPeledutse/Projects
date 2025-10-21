import io
import base64

import httpx

import pillow_heif
from PIL import Image

import streamlit as st


#adding ability to open heic images
pillow_heif.register_heif_opener()



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
    type=["png", "jpg", "jpeg", "heic", "heif"],
    help="Supported formats: PNG, JPG, JPEG, HEIC, HEIF"
)

if uploaded_file is not None:
    # Display uploaded image in a clean container
    image = Image.open(uploaded_file)
    #If the image is in HEIF format
    if image.format == 'HEIF':
        image = image.convert("RGB")
        jpeg_buffer = io.BytesIO()
        image.save(jpeg_buffer,format = "JPEG")
        image = Image.open(jpeg_buffer)
    #display the image
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
                # backend API endpoint
                api_url = "http://127.0.0.1:8000/prediction"

                # POST request with the image file
                response = httpx.post(
                    api_url,
                    json = {"image_64_base": image_64},
                    timeout = 10 
                )

                if response.status_code == 200:
                    prediction = response.json()
                    
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
