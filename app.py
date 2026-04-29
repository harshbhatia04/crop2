from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import random
import time
import httpx # Added for Groq API
from pydantic import BaseModel
from dotenv import load_dotenv

# Load keys from .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

class AIRequest(BaseModel):
    crop: str
    disease: str
    language: str = "English"

class TTSRequest(BaseModel):
    text: str
    language: str

USING_REAL_MODEL = True

print("Starting web app initialization...")
try:
    print("Importing inference module... (this may take a minute for PyTorch to load)")
    from inference import CropDiseaseInference
    print("Loading weights into model...")
    predictor = CropDiseaseInference("best_model_vision.pth", "class_symptoms.json", 17)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Model could not be loaded, using mock only: {e}")
    predictor = None
app = FastAPI(title="Crop Vision App")

# Allows CORS for frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure temp directory exists for uploads
os.makedirs("temp", exist_ok=True)

# Important: Serve static files at the root
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

MOCK_CLASSES = [
    {"crop": "Tomato", "disease": "Early Blight", "symptom": "Brown spots with concentric rings on lower leaves."},
    {"crop": "Potato", "disease": "Late Blight", "symptom": "Water-soaked, pale green spots that turn brown/purplish."},
    {"crop": "Corn", "disease": "Common Rust", "symptom": "Elongated, reddish-brown pustules on both leaf surfaces."},
    {"crop": "Tomato", "disease": "Healthy", "symptom": "Beautiful, healthy green plant! No action required."}
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = f"temp/{file.filename}"
    
    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        if USING_REAL_MODEL:
            # Use the actual VLM inference
            result = predictor.predict_image(file_path)
            return JSONResponse(content=result)
        else:
            # Simulator Mode: So we can build the UI before the 3-hour training is done!
            time.sleep(1.5) # Simulate processing time
            choice = random.choice(MOCK_CLASSES)
            return JSONResponse(content={
                "crop": choice["crop"],
                "disease": choice["disease"],
                "confidence": f"{random.uniform(85.0, 99.9):.2f}%",
                "symptoms": choice["symptom"],
                "visual_explanation": None # Placeholder until GradCAM is active
            })
    finally:
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/ask_ai")
async def ask_ai(request: AIRequest):
    prompt = f"You are a professional agronomist. A farmer has a {request.crop} plant with {request.disease}. Provide a concise, professional diagnosis summary and 3 specific, actionable steps to treat it (Organic and Chemical). IMPORTANT: You MUST write the entire response in {request.language}. Keep it under 150 words."
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7
                },
                timeout=10.0
            )
            data = response.json()
            if 'error' in data:
                return JSONResponse(content={"advice": f"Groq API Error: {data['error'].get('message', 'Unknown Error')}"}, status_code=400)
            
            if 'choices' in data:
                return JSONResponse(content={"advice": data['choices'][0]['message']['content']})
            else:
                return JSONResponse(content={"advice": f"Unexpected API Response: {str(data)}"}, status_code=500)
        except Exception as e:
            return JSONResponse(content={"advice": f"Connection Error: {str(e)}"}, status_code=500)

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    print(f"DEBUG: Sarvam TTS requested for language: {request.language}")
    lang_map = {
        "Hindi": "hi-IN", "Marathi": "mr-IN", "Telugu": "te-IN", "Tamil": "ta-IN",
        "Bengali": "bn-IN", "Kannada": "kn-IN", "Punjabi": "pa-IN", "Gujarati": "gu-IN",
        "Malayalam": "ml-IN", "Odia": "or-IN"
    }
    target_code = lang_map.get(request.language, "en-IN")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.sarvam.ai/text-to-speech",
                headers={"api-subscription-key": SARVAM_API_KEY},
                json={
                    "inputs": [request.text],
                    "target_language_code": target_code,
                    "speaker": "anushka",
                    "pitch": 0, "pace": 1.0, "loudness": 1.5,
                    "speech_sample_rate": 8000
                },
                timeout=15.0
            )
            data = response.json()
            if response.status_code != 200:
                print(f"SARVAM API ERROR: {data}")
                return JSONResponse(content={"error": f"Sarvam Error: {str(data)}"}, status_code=response.status_code)
                
            return JSONResponse(content={"audio": data['audios'][0]})
        except Exception as e:
            print(f"TTS EXCEPTION: {str(e)}")
            return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    # Runs the server locally on port 8090
    uvicorn.run(app, host="127.0.0.1", port=8090)
