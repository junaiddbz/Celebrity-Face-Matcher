import streamlit as st
import os
import pickle
import numpy as np
from PIL import Image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from mtcnn import MTCNN
from sklearn.neighbors import NearestNeighbors

# --- PAGE CONFIG ---
st.set_page_config(page_title="Celebrity Face Matcher", page_icon="😎")

st.title("😎 Celebrity Look-Alike Finder")
st.markdown("Upload a photo to find your celebrity doppelgänger using **VGGFace (ResNet50)** and **MTCNN**.")

# --- CACHED RESOURCES ---
# We use @st.cache_resource to load heavy models only once, not on every reload.
@st.cache_resource
def load_models():
    # Load VGGFace with ResNet50 architecture (weights will download on first run)
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    
    # Load MTCNN for face detection
    detector = MTCNN()
    return model, detector

@st.cache_data
def load_embeddings():
    # Load pre-computed embeddings and filenames
    # ENSURE 'embedding.pkl' and 'filenames.pkl' are in a 'PKL' folder in your repo
    feature_list = np.array(pickle.load(open('PKL/embedding.pkl', 'rb')))
    filenames = pickle.load(open('PKL/filenames.pkl', 'rb'))
    return feature_list, filenames

model, detector = load_models()
feature_list, filenames = load_embeddings()

# --- UTILITY FUNCTIONS ---
def extract_features(face_array, model):
    # Preprocess image for VGGFace (ResNet50 expects specific normalization)
    face_array = face_array.astype('float32')
    expanded_img = np.expand_dims(face_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result

def recommend(feature_list, features):
    # Use NearestNeighbors (Cosine Similarity) instead of complex clustering for speed in the web app
    neighbors = NearestNeighbors(n_neighbors=1, algorithm='brute', metric='cosine')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices[0][0]

# --- MAIN APP LOGIC ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=300)
    
    if st.button('Find Match'):
        with st.spinner('Analyzing facial features...'):
            try:
                # Convert to numpy array for MTCNN
                img_array = np.array(image)
                
                # 1. Detect Face
                results = detector.detect_faces(img_array)
                
                if not results:
                    st.error("No face detected! Please try a clearer photo.")
                else:
                    # Extract the bounding box of the first face
                    x, y, width, height = results[0]['box']
                    face = img_array[y:y+height, x:x+width]
                    
                    # 2. Resize to 224x224 (VGGFace Input)
                    face_image = Image.fromarray(face).resize((224, 224))
                    
                    # 3. Extract Features
                    face_array = np.asarray(face_image)
                    features = extract_features(face_array, model)
                    
                    # 4. Find Match
                    index_pos = recommend(feature_list, features)
                    predicted_actor = " ".join(filenames[index_pos].split('\\')[-1].split('_')) # Adjust split based on your filename format
                    
                    # 5. Display Result
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.header("Your Face")
                        st.image(face_image, width=200)
                        
                    with col2:
                        st.header("Celebrity Match")
                        st.image(filenames[index_pos], width=200, caption=predicted_actor)
                        
                    st.success(f"Match Found: {predicted_actor}")

            except Exception as e:
                st.error(f"Error processing image: {e}")