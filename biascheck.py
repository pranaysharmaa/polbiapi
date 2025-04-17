from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Pre-load the model here, before FastAPI gets any requests
tokenizer = AutoTokenizer.from_pretrained("cajcodes/DistilBERT-PoliticalBias")
model = AutoModelForSequenceClassification.from_pretrained("cajcodes/DistilBERT-PoliticalBias")
model.eval()  # Set to eval mode to improve performance

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow requests from any origin (or specify your extension's domain if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify a list of allowed origins, e.g., ["https://your-extension-url.com"]
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define the request model
class TextInput(BaseModel):
    text: str

# POST route to predict bias
@app.post("/predict_bias")
def predict_bias(input: TextInput):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    # Map predicted class to label
    label_map = {
        0: "Left",
        1: "Center",
        2: "Right"
    }

    return {
        "text": input.text,
        "predicted_class": predicted_class,
        "bias_label": label_map.get(predicted_class, "Unknown")
    }

# Root GET route to check API status
@app.get("/")
def read_root():
    return {"message": "BiasCheck API is running. Use POST /predict_bias with JSON input."}

# Optional: GET route for testing directly from browser
@app.get("/predict_bias")
def demo_predict_bias():
    sample_input = TextInput(text="The government's new initiative is a remarkable step forward.")
    return predict_bias(sample_input)
