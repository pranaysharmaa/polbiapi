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

# Add CORS middleware to allow requests from any origin (important for your Chrome extension)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
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
    # Tokenize the input text
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():  # Disable gradient computation for inference
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()  # Get the predicted class index

    # Map the predicted class to the corresponding label (bias type)
    label_map = {
        0: "Left",
        1: "Center",
        2: "Right"
    }

    # Return the prediction
    return {
        "text": input.text,
        "predicted_class": predicted_class,
        "bias_label": label_map.get(predicted_class, "Unknown")
    }

# Root GET route to check API status
@app.get("/")
def read_root():
    return {"message": "BiasCheck API is running. Use POST /predict_bias with JSON input."}
