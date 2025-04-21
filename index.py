import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
from cotton_disease_detection import CottonDiseaseClassifier, get_severity
import os
import requests

st.set_page_config(page_title="Cotton Disease Detection", layout="wide")

# 🎨 Custom Styling: Sailor Blue & Mint
dark_blue = "#00203FFF"
mint = "#ADEFD1FF"

st.markdown(f"""
    <style>
        html, body, [data-testid="stApp"] {{
            background-color: {dark_blue};
            color: {mint};
        }}

        .block-container {{
            background-color: {dark_blue} !important;
            color: {mint};
        }}

        h1, h2, h3, h4, h5, h6, label, p, span, .stTextInput, .stTextArea {{
            color: {mint} !important;
        }}

        .title {{
            color: {mint};
            font-size: 2.5em;
            text-align: center;
        }}

        section[data-testid="stSidebar"] {{
            background-color: #011627 !important;
            border-right: 2px solid {mint};
        }}

        .stButton>button {{
            background-color: {mint};
            color: {dark_blue};
            border-radius: 10px;
            font-weight: bold;
        }}

        .stButton>button:hover {{
            background-color: #9ad9c2;
        }}

        .stTextInput input, .stTextArea textarea {{
            background-color: #01303F;
            color: {mint} !important;
            border: 1px solid {mint};
        }}

        .stAlert {{
            background-color: #01303F !important;
            color: {mint} !important;
        }}

        .stInfo {{
            background-color: #01303F !important;
            color: {mint} !important;
            border-left: 5px solid {mint};
        }}

        .stSuccess {{
            background-color: #014F4F !important;
            color: #00ffaa !important;
        }}

        .stWarning {{
            background-color: #604c1e !important;
            color: #ffe066 !important;
        }}

        .stError {{
            background-color: #6d3a3a !important;
            color: #ff6b6b !important;
        }}

        .stChatMessage {{
            background-color: #01303F;
            color: {mint} !important;
            border-radius: 10px;
            padding: 10px;
        }}

    </style>
""", unsafe_allow_html=True)


# -------------------- PAGE: Cotton Disease Detection --------------------
def load_model():
    try:
        checkpoint = torch.load('best_model.pth', map_location=torch.device('cpu'))
        model = CottonDiseaseClassifier(num_classes=len(checkpoint['class_names']))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model, checkpoint['class_names']
    except FileNotFoundError:
        st.error("Model file 'best_model.pth' not found. Please train the model first.")
        return None, None

def predict_image(model, image, class_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)
        
    return class_names[predicted], confidence.item()

def cotton_disease_page():
    st.markdown("<h1 class='title'>Cotton Disease Detection System</h1>", unsafe_allow_html=True)
    
    model, class_names = load_model()
    if model is None:
        return

    st.write("### Upload an image of a cotton plant leaf")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        with col1:
            st.image(image, caption='Uploaded Image', use_container_width=True)
        
        predicted_class, confidence = predict_image(model, image, class_names)
        severity = get_severity(confidence)
        
        with col2:
            st.write("### Results")
            st.write("**Detected Disease:**")
            st.info(predicted_class)

            st.write("**Disease Severity:**")
            if severity == "Mild":
                st.success(f"🟢 {severity}")
            elif severity == "Moderate":
                st.warning(f"🟡 {severity}")
            else:
                st.error(f"🔴 {severity}")
            
            st.write("### Recommendations")
            recommendations = {
                "Healthy": "Continue regular maintenance and monitoring.",
                "Bacterial Blight": "1. Remove infected plants\n2. Apply copper-based bactericides\n3. Improve air circulation",
                "Curl Virus": "1. Control whitefly population\n2. Remove infected plants\n3. Use resistant varieties",
                "Fussarium Wilt": "1. Crop rotation\n2. Use resistant varieties\n3. Soil solarization",
                "Target spot": "1. Apply fungicides\n2. Improve drainage\n3. Reduce leaf wetness",
                "Aphids": "1. Use insecticidal soaps\n2. Introduce natural predators\n3. Remove infected parts",
                "Army worm": "1. Apply appropriate insecticides\n2. Monitor field regularly\n3. Use pheromone traps",
                "Powdery Mildew": "1. Apply fungicides\n2. Improve air circulation\n3. Avoid overhead irrigation"
            }
            
            if predicted_class in recommendations:
                st.info(recommendations[predicted_class])
    
    st.sidebar.write("### About the Model")
    st.sidebar.write("""
    This application uses a deep learning model based on ResNet50 to detect diseases in cotton plants.
    """)
    for class_name in class_names:
        st.sidebar.write(f"- {class_name}")
    st.sidebar.write("### Severity Legend")
    st.sidebar.write("🟢 **Mild** (< 50%)  \n🟡 **Moderate** (50-80%)  \n🔴 **Severe** (> 80%)")

# -------------------- PAGE: Chatbot --------------------
def chatbot_page():
    st.title("MongoDB as Vector-Store 🙂")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Ask your Question!!!"):
        st.session_state.messages.append({"role": "User", "content": query})
        with st.chat_message("User"):
            st.markdown(query)
        
        try:
            response = requests.post("http://localhost:8000/api", json={"input": {"input": query}})
            result = response.json()['response']
        except requests.exceptions.RequestException as e:
            result = f"An error occurred: {e}"
        
        with st.chat_message("Bot", avatar="🤖"):
            st.markdown(result)
        st.session_state.messages.append({"role": "Bot", "content": result})

# -------------------- MAIN --------------------
def main():
    # Inject custom CSS for Sailor Blue and Mint color scheme
    try:
        with open("style.css") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("style.css file not found. Using default Streamlit theme.")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Cotton Disease Detection", "Chatbot"])
    
    if page == "Cotton Disease Detection":
        cotton_disease_page()
    elif page == "Chatbot":
        chatbot_page()

if __name__ == "__main__":
    main()