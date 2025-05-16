import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from fpdf import FPDF
import os
import base64
from io import BytesIO
from datetime import datetime
from openai import OpenAI
import unicodedata

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ========== OpenAI Setup ==========
client = OpenAI()  # Ensure you have set OPENAI_API_KEY env variable or set here

# ========== Streamlit Config ==========
st.set_page_config(layout="wide")

# ========== Load Model ==========
@st.cache_resource
def load_model_custom():
    # return tf.keras.models.load_model('inception_model.h5')
    return tf.keras.models.load_model('best_model.h5')

model = load_model_custom()
last_conv_layer_name = "mixed10"
# last_conv_layer_name = "top_conv"
class_labels = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# ========== Utils ==========
def preprocess_image(image: Image.Image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    if image.shape[-1] == 4:  # Handle RGBA
        image = image[..., :3]
    return np.expand_dims(image, axis=0)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(image, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image.width, image.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    image_np = np.array(image)
    superimposed_img = heatmap_color * alpha + image_np
    return Image.fromarray(np.uint8(superimposed_img))

def sanitize_text(text):
    # Replace curly quotes and other problematic Unicode with safe ASCII
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

def generate_pdf_report(patient_info, prediction, description, precautions, original_img, gradcam_img, output_path="report.pdf"):
    description = sanitize_text(description)
    precautions = sanitize_text(precautions)
    pdf = FPDF()
    pdf.add_page()

    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Tumor Classification Report", ln=True, align='C')
    pdf.set_font("Arial", '', 12)
    pdf.ln(10)
    for key, value in patient_info.items():
        pdf.cell(0, 10, f"{key}: {value}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Prediction: {prediction}", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 10, f"\nDescription:\n{description}")
    pdf.multi_cell(0, 10, f"\nPrecautions:\n{precautions}")

    # Add images
    def save_img(img, filename):
        temp_path = f"./tmp/{filename}"
        img.save(temp_path)
        return temp_path
    
    ori_path = save_img(original_img, "original.png")
    grad_path = save_img(gradcam_img, "gradcam.png")

    image_width = 90
    image_height = 60  # adjust based on aspect ratio
    spacing = 5
    page_height = 297  # A4 size in mm
    bottom_margin = 10

    # Get current vertical position
    current_y = pdf.get_y()

    # Ensure there is enough space for images
    if current_y + image_height + bottom_margin > page_height:
        pdf.add_page()
        current_y = pdf.get_y()

    # Insert both images on the same row
    pdf.image(ori_path, x=10, y=current_y + spacing, w=image_width, h=image_height)
    pdf.image(grad_path, x=110, y=current_y + spacing, w=image_width, h=image_height)

    # Optional: move cursor below the images if you want to add more content
    pdf.set_y(current_y + image_height + spacing + 10)

    pdf.output(output_path)
    return output_path

def pil_to_base64(pil_image, format='JPEG'):
    """
    Converts a PIL Image to a base64 encoded string.

    Args:
        pil_image (PIL.Image.Image): The PIL Image to encode.
        format (str): Format to save the image in (e.g., 'PNG', 'JPEG').

    Returns:
        str: Base64 encoded string of the image.
    """
    buffered = BytesIO()
    pil_image.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    encoded_str = base64.b64encode(img_bytes).decode('utf-8')
    return encoded_str

def generate_llm_prompt(class_label, patient_info):
    prompt = f"""
You are a medical assistant. A CNN model has analyzed an MRI scan and predicted the tumor type as **{class_label}**.
Patient information:
- Age: {patient_info.get('Age')}
- Gender: {patient_info.get('Gender')}
- Symptoms: {patient_info.get('Symptoms')}
As a medical expert, please examine the patient’s condition by ﬁrst identifying any abnormal. Next, critically analyze there their impact, and clearly state ﬁnal diagnosis regarding what might be causing the clinical deterioration. Finally give a brief description.
Provide a clear medical description of the predicted tumor type, followed by helpful precautions for the patient.

Format:
Description: ...
Precautions: ...
"""
    return prompt

# ========== App UI ==========
st.title("🧠 Brain Tumor Classifier with Medical Assistant")

tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload", "💡Prediction", "📄 Report", "🤖 AI Assistant"])

# === Upload Tab ===
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Name", placeholder="Name")
        age = st.number_input("Age", min_value=0, max_value=120)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    with col2:
        symptoms = st.text_area("Symptoms (optional)", placeholder="E.g. Headache, vision issues...")
        uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        submitted = st.button("Submit")

with tab2:
    if uploaded_file and submitted:
        image = Image.open(uploaded_file).convert("RGB")

        img_array = preprocess_image(image)
        try:
            prediction = model.predict(img_array)
        except Exception as e:
            try:
                prediction = model.predict({'input_layer_2': img_array})
            except Exception as e2:
                st.error(f"Prediction failed: {e}\nAlso tried dict input: {e2}")
                st.stop()
        class_index = np.argmax(prediction)
        class_label = class_labels[class_index]

        st.markdown(f"### 🧠 Model Prediction: `{class_label}`")

        # Grad-CAM
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        heatmap_img = overlay_heatmap(image, heatmap)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original MRI", use_container_width=True)
        with col2:
            st.image(heatmap_img, caption="Grad-CAM", use_container_width=True)

with tab3:
    if uploaded_file and submitted:
        # GPT Explanation
        patient_info = {"Name": name, "Age": age, "Gender": gender, "Symptoms": symptoms}
        prompt = generate_llm_prompt(class_label, patient_info)
        base64_image = pil_to_base64(image)
        with st.spinner("💬 Generating description and precautions using GPT..."):
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=[
                        {"role": "user", "content": prompt},
                        {"role": "user", "content": [{
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{base64_image}",
                            }],
                        },
                    ],
                temperature=0.5,
            )
            result = response.output_text

        if "Precautions:" in result:
            parts = result.split("Precautions:")
            desc = parts[0].replace("Description:", "").strip()
            precautions = parts[1].strip()
        else:
            desc = result.strip()
            precautions = "Not provided."

        st.markdown("### 📝 Medical Description")
        st.write(desc)

        st.markdown("### ✅ Suggested Precautions")
        st.write(precautions)

        # Save Grad-CAM image for report
        gradcam_path = "gradcam_temp.jpg"
        heatmap_img.save(gradcam_path)
        pdf_path = generate_pdf_report(patient_info, class_label, desc, precautions, image, heatmap_img)
        
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()

        st.download_button(
            label="⬇️ Download Report",
            data=pdf_data,
            file_name="tumor_report.pdf",
            on_click="ignore"
        )

with tab4:
    if uploaded_file and submitted:
        # Chat Interface
        st.markdown("### 💬 Ask the Assistant a Follow-up Question")
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [{"role": "system", "content": f"You are a medical assistant. The tumor was classified as {class_label}. {desc}"}]

        # for msg in st.session_state.chat_history:
        #     if msg["role"] != "system":
        #         st.chat_message(msg["role"]).write(msg["content"])

        user_query = st.chat_input("Ask a question about the diagnosis, precautions, or next steps:")
        # if user_query:
        #     st.session_state.chat_history.append({"role": "user", "content": user_query})
        #     with st.spinner("Typing..."):
        #         chat_resp = client.chat.completions.create(
        #             model="gpt-4.1-mini",
        #             messages=st.session_state.chat_history,
        #             temperature=0.6,
        #         )
        #         reply = chat_resp.choices[0].message.content
        #         st.session_state.chat_history.append({"role": "assistant", "content": reply})
        #         st.chat_message("assistant").write(reply)
        if user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)
            
            with st.chat_message("assistant"):
                stream = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    temperature=0.6,
                    messages=[
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.chat_history
                    ],
                    stream=True,
                )
                response = st.write_stream(stream)
            # st.session_state.chat_history.append({"role": "assistant", "content": response})

            # with st.spinner("Typing..."):
            #     chat_resp = client.chat.completions.create(
            #         model="gpt-4.1-mini",
            #         messages=st.session_state.chat_history,
            #         temperature=0.6,
            #     )
            #     reply = chat_resp.choices[0].message.content
            #     st.session_state.chat_history.append({"role": "assistant", "content": reply})
            #     st.chat_message("assistant").write(reply)