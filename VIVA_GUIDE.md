# 🎓 Viva Preparation Guide: CropVision AI

This document explains exactly how each file in your project works. Use this to study for your viva and answer the examiner's questions confidently.

---

## 🏗️ Project Architecture Overview
This is a **Full-Stack AI Web Application**. 
- **Backend**: Python (FastAPI) handles the image processing and AI model.
- **Frontend**: HTML/CSS/JS (Vanilla) provides a modern "Neo-Brutalist" user interface.
- **AI Core**: PyTorch (EfficientNet-B3) identifies the disease, and Grad-CAM provides visual explanations.

---

## 📂 File-by-File Breakdown

### 1. `app.py` (The Heart of the Server)
*   **Purpose**: This is your backend server built using **FastAPI**.
*   **Key Logic**:
    *   It defines the `/predict` endpoint that the website talks to.
    *   It initializes the `Predictor` and handles the model weights.
    *   It serves the static files (HTML/CSS/JS) so you can see the website in your browser.
    *   **Examiner Question**: "How does the frontend talk to the backend?"
    *   **Your Answer**: "The frontend sends an image via an HTTP POST request to the `/predict` route in `app.py`, which then calls the AI model."

### 2. `inference.py` (The AI Engine)
*   **Purpose**: This is where the actual "thinking" happens.
*   **Key Logic**:
    *   **Model Loading**: It loads the `EfficientNet-B3` architecture and the trained weights.
    *   **Prediction**: It takes an image, transforms it (resizes to 300x300), and runs it through the neural network.
    *   **Grad-CAM**: It calculates the "gradients" of the last convolutional layer to create the heatmap visualization.
    *   **Examiner Question**: "Why did you use EfficientNet-B3?"
    *   **Your Answer**: "It is a highly efficient convolutional neural network that provides state-of-the-art accuracy with fewer parameters compared to older models like ResNet."

### 3. `static/index.html` (The Structure)
*   **Purpose**: Defines the layout of your website.
*   **Key Logic**:
    *   Contains the "Upload" card and the "Result" card.
    *   Uses semantic HTML5 and includes the Google Font 'Inter' for a premium feel.

### 4. `static/style.css` (The Look)
*   **Purpose**: Implements the **Neo-Brutalist** design system.
*   **Key Logic**:
    *   Uses bold colors, thick black borders, and heavy shadows.
    *   Includes media queries to make the app responsive on different screen sizes.

### 5. `static/script.js` (The Interaction)
*   **Purpose**: Logic for the browser side.
*   **Key Logic**:
    *   **Drag & Drop**: Handles file uploads via dragging or clicking.
    *   **Dynamic UI**: Shows the photo preview instantly and updates the diagnosis text without refreshing the page.
    *   **Cache Busting**: Adds a timestamp to the Grad-CAM image URL so the browser doesn't show an old heatmap.

### 6. `class_symptoms.json` (The Database)
*   **Purpose**: A mapping file that translates technical class names (like `Tomato_Early_Blight`) into human-readable symptoms and action steps.

### 7. `model_training_vision_only.py` (The Training Lab)
*   **Purpose**: The script used to train the final model.
*   **Key Logic**: Uses **Transfer Learning** (starting with a pre-trained model and fine-tuning it on your crop dataset).

---

## 🧠 Key AI Concepts to Know

### What is Grad-CAM?
It stands for **Gradient-weighted Class Activation Mapping**. It’s a technique for making AI "explainable" by highlighting which parts of an image the model used to make its prediction. Red areas mean "high importance."

### What is Transfer Learning?
Instead of training a model from scratch, we used a model that already "knows" how to see (trained on ImageNet) and taught it specifically about crop diseases.

### Why not VLM?
We experimented with a **Vision-Language Model (VLM)** in `model_training.py`, but chose a pure Vision model for the final app because it is faster, more robust against "label leakage," and allows for better visual explanations with Grad-CAM.

---

**💡 Pro Tip for your Viva**: If the examiner asks why you did something a certain way, always mention **"User Experience"** and **"Model Explainability."** They love those terms!
