import streamlit as st
import requests
import base64
from PIL import Image, UnidentifiedImageError
import io

# ----------------- CONFIG -----------------
API_URL = "https://ijtoaz6584.execute-api.eu-north-1.amazonaws.com/test/predict"
ALLOWED_SPECIES = [
    'Asian Green Bee-Eater', 'Brown-Headed Barbet', 'Cattle Egret', 'Common Kingfisher',
    'Common Myna', 'Common Rosefinch', 'Common Tailorbird', 'Coppersmith Barbet',
    'Forest Wagtail', 'Gray Wagtail', 'Hoopoe', 'House Crow', 'Indian Grey Hornbill',
    'Indian Peacock', 'Indian Pitta', 'Indian Roller', 'Jungle Babbler',
    'Northern Lapwing', 'Red-Wattled Lapwing', 'Ruddy Shelduck', 'Rufous Treepie',
    'Sarus Crane', 'White Wagtail', 'White-Breasted Kingfisher', 'White-Breasted Waterhen'
]

st.set_page_config(
    page_title="Bird Classifier (Capstone1-MLZ)",
    page_icon="🦜",
    layout="wide"
)



# ----------------- APP BODY -----------------
with st.container():
    st.title("🦜 Sapient Sapiens's Indian Bird Species Classifier")
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
        - Accepted formats: **PNG, JPEG, WEBP**.  
        - The model will predict the species out of the following list:  
        """)
        st.write("✅ Allowed Bird Species:")
        st.markdown(
            "<div style='max-height: 500px; width: 85%; overflow-y: auto; border: 1px solid #ddd; "
            "padding: 10px; border-radius: 8px; background-color: #f9f9f9;'>"
            + "<br>".join(ALLOWED_SPECIES) +
            "</div>",
            unsafe_allow_html=True
        )
    

    with col2:
        st.subheader("📤 Upload Image & Get Prediction")

        uploaded_file = st.file_uploader("Upload a bird image", type=["jpg", "jpeg", "png", "webp"])
        
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

        if predict_button:
            if not uploaded_file:
                result_box.error("⚠️ Please upload an image before clicking Predict.")
            else:
                try:
                    # Prepare image
                    '''img = Image.open(uploaded_file)
                    img = img.resize((299, 299)).convert("RGB")
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG")
                    img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")'''
                    # although payload is heavier, I find this way more reliant as per test results!!
                    with open(uploaded_file, "rb") as f_in:
                        img_b64 = base64.b64encode(f_in.read()).decode("utf-8")

                    # Send request to Lambda
                    data = {"image_base64": img_b64}
                    response = requests.post(API_URL, json=data)
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
                    st.image(uploaded_file, caption="Uploaded Bird Image", use_container_width=True)

                except UnidentifiedImageError:
                    result_box.error("❌ Unsupported image format. Please upload JPG/PNG.")
                except Exception as e:
                    result_box.error(f"⚠️ Unexpected error: {e}")
