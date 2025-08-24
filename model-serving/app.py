import streamlit as st
import requests
import base64
from PIL import Image, UnidentifiedImageError


# ----------------- CONFIG -----------------

# Load API key from Streamlit secrets safely
try:
  API_KEY = st.secrets["api"]["key"]
except KeyError:
  st.error("❌🔑 API key not found. Please configure Streamlit secrets with [api] key.")
  st.stop()

API_URL = "https://vnj365geif.execute-api.eu-north-1.amazonaws.com/beta/predict"

ALLOWED_SPECIES = [
    'Asian Green Bee-Eater', 'Brown-Headed Barbet', 'Cattle Egret', 'Common Kingfisher',
    'Common Myna', 'Common Rosefinch', 'Common Tailorbird', 'Coppersmith Barbet',
    'Forest Wagtail', 'Gray Wagtail', 'Hoopoe', 'House Crow', 'Indian Grey Hornbill',
    'Indian Peacock', 'Indian Pitta', 'Indian Roller', 'Jungle Babbler',
    'Northern Lapwing', 'Red-Wattled Lapwing', 'Ruddy Shelduck', 'Rufous Treepie',
    'Sarus Crane', 'White Wagtail', 'White-Breasted Kingfisher', 'White-Breasted Waterhen'
]

st.set_page_config(
    page_title="Indian Bird Classifier",
    page_icon="🦜",
    layout="wide"
)



# ----------------- APP BODY -----------------
with st.container():
    st.title("🦜 Indian Bird Species Classifier")
    st.markdown(
            "<hr style='text-align: center; color: gray; font-size: 24px;'>"
             "<b>©Authored by Siddhartha Gogoi</b>"
            "</hr><br><br/>",
            unsafe_allow_html=True
    )
  
    st.markdown("Upload a bird image and let the deployed ML model identify its species.")

    # ----------------- LAYOUT -----------------
    col1, col2 = st.columns([1, 2])

    # ---- LEFT COLUMN: Instructions ----
    with col1:
        st.subheader("📋 Instructions")
        st.markdown("""
        - Upload an image of a bird.  
        - Accepted formats: **PNG, JPEG, WEBP, AVIF, TIFF, BMP**.  
        - The model will predict the species out of the following list:  
        """)
        st.write("✅ Allowed Bird Species:")
        st.markdown(
            "<div style='max-height: 600px; width: 320px; overflow-y: auto; border: 1px solid #ddd; "
            "padding: 10px; border-radius: 8px; background-color: #f9f9f9; font-size: 14px;'>"
            + "<br><b>".join(ALLOWED_SPECIES) + "</b></div>",

            unsafe_allow_html=True
        )

    

    with col2:
        st.subheader("📤 Upload Image & Get Prediction")

        uploaded_file = st.file_uploader("Upload a bird image", type=["jpg", "jpeg", "png", "webp", "avif", "bmp", ])
        
        # Confidence threshold slider
        confidence_threshold = st.slider(
            "Select Confidence Threshold (%)",
            min_value=70,
            max_value=100,
            value=90,
            step=1,
            help="Predictions below this confidence will be marked as 'Not Sure'."
        )

        predict_button = st.button("🔮 Predict")
        result_box = st.empty()


        def show_image_preview(uploaded):
            """Preview without opening via PIL. Use HTML data URI for AVIF/HEIC/HEIF."""
            mime = (uploaded.type or "").lower()
            st.caption("Preview of uploaded image:")
            # Formats PIL/Streamlit can usually render directly
            pil_friendly = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp", "image/tiff"}
            if mime in pil_friendly:
                uploaded.seek(0)
                st.image(uploaded, use_container_width=True)
            else:
                # e.g., image/avif, image/heic, image/heif — render via browser
                b64 = base64.b64encode(uploaded.getbuffer()).decode("utf-8")
                st.markdown(
                    f"<img src='data:{mime};base64,{b64}' style='max-width:100%; border-radius:8px;'/>",
                    unsafe_allow_html=True
                )
            uploaded.seek(0)  # keep pointer sane for any later reads


        if predict_button:
            if not uploaded_file:
                result_box.error("⚠️ Please upload an image before clicking Predict.")
            else:
                try:
                    # Prepare image

                    # ---- NO IMAGE OPENING: read raw bytes and base64 encode (test-script style) ----
                    file_bytes_view = uploaded_file.getbuffer()     # zero-copy memoryview
                    img_b64 = base64.b64encode(file_bytes_view).decode("utf-8")

                    # Send request to Lambda
                    data = {"image_base64": img_b64}

                    # change for incorporating API KEY
                    headers = {"x-api-key": API_KEY}
                    response = requests.post(API_URL, headers=headers, json=data)
                    
                    if response.status_code == 200:      
                        result = response.json()
                        predicted_class = result.get("predicted_class")
                        probability = result.get("probability", 0)
                        # Confidence level as percentage
                        confidence_percentage = probability * 100
                        # Show confidence progress bar
                        progress_color = "green" if confidence_percentage >= 90 else "orange" if confidence_percentage >= 80 else "red"
                        st.markdown(
                            f"""
                            <div style='width:100%;background-color:#ddd;border-radius:8px;'>
                                <div style='width:{confidence_percentage:.2f}%;background-color:{progress_color};
                                            padding:4px;border-radius:8px;text-align:center;color:white;'>
                                    {confidence_percentage:.2f}%
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        # Decide response based on slider
                        if confidence_percentage < confidence_threshold:
                            display_text = "🤔 The model is not confident enough to identify this bird."
                        else:
                            display_text = (
                                f"✅ The bird appears to be a **{predicted_class}** "
                                f"with a confidence of **{confidence_percentage:.2f}%**."
                            )

                        result_box.info(display_text)

                        # Show uploaded image
                        show_image_preview(uploaded_file)
                        #st.image(uploaded_file, caption="Uploaded Bird Image", use_container_width=True)

                    else:
                      try:
                        error_msg = response.json()  # Try parsing JSON
                      except ValueError:
                        error_msg = response.text   # Fallback to raw text if not JSON
                      result_box.error(f"❌🛠️🔑 API Error {response.status_code}: {error_msg}")


                except UnidentifiedImageError:
                    result_box.error("❌🖼️ Issue in image format for this image. Please explictly convert to JPG/PNG before uploading!.")
                except Exception as e:
                    result_box.error(f"⚠️ Unexpected error: {e}")
