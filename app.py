import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import cv2

# Page configuration
st.set_page_config(
    page_title="CIFAR-10 Live Classifier",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .confidence-high {
        color: #2ecc71;
        font-weight: bold;
    }
    .confidence-medium {
        color: #f39c12;
        font-weight: bold;
    }
    .confidence-low {
        color: #e74c3c;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# Emoji mapping
EMOJI_MAP = {
    'airplane': '🛫',
    'automobile': '🚗',
    'bird': '🐦',
    'cat': '🐱',
    'deer': '🦌',
    'dog': '🐕',
    'frog': '🐸',
    'horse': '🐴',
    'ship': '⛴️',
    'truck': '🚛'
}

@st.cache_resource
def load_trained_model():
    """
    Load the trained model (cached to avoid reloading)
    """
    try:
        model = load_model('cifar10_model.h5')
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'cifar10_model.h5' not found!")
        st.info("📌 Please run the Jupyter notebook first to train and save the model:")
        st.code("python -m jupyter notebook train_model.ipynb", language="bash")
        st.stop()

def preprocess_image(image):
    """
    Preprocess image for model prediction
    """
    # Ensure image is in RGB mode
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA
                image = Image.fromarray(image)
                image = image.convert('RGB')
            elif image.shape[2] == 3:  # BGR (from OpenCV)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
    
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to 32x32 (CIFAR-10 image size)
    image_resized = image.resize((32, 32))
    
    # Convert to numpy array and normalize
    image_array = np.array(image_resized).astype('float32') / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array, image_resized, image

def predict_image(model, image_array):
    """
    Make prediction on the image
    """
    predictions = model.predict(image_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    return predicted_class_idx, confidence, predictions[0]

def display_prediction_results(predicted_class_idx, confidence, all_predictions, image):
    """
    Display detailed prediction results
    """
    predicted_class = CLASS_NAMES[predicted_class_idx]
    emoji = EMOJI_MAP.get(predicted_class, '📊')
    
    # Main prediction display
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="prediction-box">
            <h2>{emoji} {predicted_class.upper()}</h2>
            <h3>Confidence: {confidence*100:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.write("")
        st.write("")
        if confidence > 0.8:
            st.success("✓ High Confidence", icon="✅")
        elif confidence > 0.5:
            st.warning("⚠ Medium Confidence", icon="⚠️")
        else:
            st.error("✗ Low Confidence", icon="❌")
    
    with col3:
        st.write("")
        st.write("")
        st.info(f"📊 Top Prediction", icon="ℹ️")
    
    # Bar chart of all predictions
    st.subheader("📈 All Class Probabilities")
    
    prediction_data = {
        CLASS_NAMES[i]: float(all_predictions[i]) 
        for i in range(len(CLASS_NAMES))
    }
    
    # Sort by confidence
    sorted_predictions = dict(sorted(
        prediction_data.items(), 
        key=lambda x: x[1], 
        reverse=True
    ))
    
    # Bar chart
    st.bar_chart(sorted_predictions)
    
    # Detailed table
    st.subheader("🔍 Detailed Scores")
    
    table_data = []
    for idx, (class_name, score) in enumerate(sorted_predictions.items(), 1):
        emoji = EMOJI_MAP.get(class_name, '📊')
        table_data.append({
            "#": idx,
            "Class": f"{emoji} {class_name.capitalize()}",
            "Confidence": f"{score*100:.2f}%"
        })
    
    st.table(table_data)

def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1>🎥 CIFAR-10 Live Image Classifier</h1>
        <p>Real-time classification with live camera or image upload</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_trained_model()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        
        input_mode = st.radio(
            "Select Input Mode:",
            options=["📸 Camera", "📤 Upload Image"],
            help="Choose how to provide input to the classifier"
        )
        
        st.divider()
        
        st.header("📋 About Model")
        st.info("""
        **Model Details:**
        - Architecture: CNN with 3 convolutional blocks
        - Parameters: 1.2M
        - Dataset: CIFAR-10
        - Test Accuracy: ~92%
        - Input Size: 32×32 pixels
        """)
        
        st.header("💡 How to Use")
        st.markdown("""
        **Camera Mode:**
        1. Click "Take Picture"
        2. Capture an image
        3. Get instant predictions
        
        **Upload Mode:**
        1. Select an image file
        2. View prediction results
        3. Check confidence scores
        """)
        
        st.header("📚 Supported Classes")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            🛫 Airplane
            🚗 Automobile
            🐦 Bird
            🐱 Cat
            🦌 Deer
            """)
        with col2:
            st.markdown("""
            🐕 Dog
            🐸 Frog
            🐴 Horse
            ⛴️ Ship
            🚛 Truck
            """)
    
    # Main content based on selected mode
    if input_mode == "📸 Camera":
        st.subheader("📸 Live Camera Feed")
        
        col_camera, col_info = st.columns([2, 1])
        
        with col_camera:
            picture = st.camera_input(
                "Take a picture",
                label_visibility="collapsed"
            )
            
            if picture is not None:
                # Convert camera input to image
                image = Image.open(picture)
                
                with st.spinner('🔍 Processing and analyzing...'):
                    # Preprocess
                    image_array, image_resized, original_image = preprocess_image(image)
                    
                    # Predict
                    predicted_class_idx, confidence, all_predictions = predict_image(
                        model, 
                        image_array
                    )
                
                st.success("✓ Prediction complete!")
                
                # Display prediction results
                display_prediction_results(
                    predicted_class_idx, 
                    confidence, 
                    all_predictions,
                    image
                )
                
                # Display processed image
                with st.expander("👀 View Processed Image (32×32)"):
                    st.image(image_resized, caption="Resized for Model Input")
        
        with col_info:
            st.info("""
            **📌 Tips for Better Results:**
            
            - Use clear lighting
            - Center the object
            - Avoid shadows
            - Keep objects simple
            - Use CIFAR-10 classes
            """)
    
    elif input_mode == "📤 Upload Image":
        st.subheader("📤 Upload an Image")
        
        col_upload, col_info = st.columns([2, 1])
        
        with col_upload:
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp'],
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                # Load image
                image = Image.open(uploaded_file)
                
                # Display original image
                st.image(image, caption="Original Image", use_column_width=True)
                
                st.divider()
                
                # Preprocess and predict
                with st.spinner('🔍 Analyzing image...'):
                    image_array, image_resized, _ = preprocess_image(image)
                    predicted_class_idx, confidence, all_predictions = predict_image(
                        model, 
                        image_array
                    )
                
                st.success("✓ Prediction complete!")
                
                # Display results
                display_prediction_results(
                    predicted_class_idx, 
                    confidence, 
                    all_predictions,
                    image
                )
                
                # Display resized image
                with st.expander("👀 View Processed Image (32×32)"):
                    st.image(image_resized, caption="Resized for Model Input")
            
            else:
                st.info("👆 Upload an image to get started!")
        
        with col_info:
            st.info("""
            **📌 Supported Formats:**
            - JPG / JPEG
            - PNG
            - BMP
            - GIF
            - WebP
            
            **📌 Tips:**
            - Use clear images
            - Optimal size: 100x100px or larger
            - CIFAR-10 class objects
            """)

if __name__ == "__main__":
    main()