import torch
import litserve as ls
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from underthesea import word_tokenize
from huggingface_hub import login

# ----------- Load model & pipeline -----------
tokenizer = AutoTokenizer.from_pretrained("Khoa/kompa-check-ads-0725", use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained("Khoa/kompa-check-ads-0725")

device_id = 0 if torch.cuda.is_available() else -1
if device_id >= 0:
    model = model.to(f"cuda:{device_id}")

text_classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=device_id  # device pipeline
)

# ----------- Preprocess & Predict -----------
def preprocess_text(text: str) -> str:
    text = text.lower()
    tokens = word_tokenize(text)
    return " ".join(tokens)

def predict_ads(text: str) -> bool:
    if not text or not text.strip():
        raise ValueError("Input text must not be empty.")
    processed_text = preprocess_text(text)
    result = text_classifier(processed_text, truncation=True, max_length=100)[0]
    label_id = int(result["label"].split("_")[-1]) if "label" in result["label"].lower() else 0
    return label_id == 1

# ----------- LitServe Handler -----------
class AdsHandler(ls.LitAPI):
    def setup(self, device):
        
        pass

    def predict(self, request: dict) -> dict:
        merged = dict(request)
        try:
            content = merged.get("content", "")
            is_ads = predict_ads(content)
            if is_ads:
                merged["label"] = "Rao vặt"
                merged["label_id"] = "68898a3c16a3634d83338269"
                merged["sentiment"] = "Neutral"
            else:
                merged["label"] = None
                merged["label_id"] = None
                merged["sentiment"] = None
        except Exception as e:
            print(f"⚠️ Error in predict_ads for item {merged.get('id')}: {e}")
            merged["label"] = None
            merged["label_id"] = None
            merged["sentiment"] = None
        return merged

if __name__ == "__main__":
    server = ls.LitServer(
        AdsHandler(),
        accelerator="gpu",                     
        devices=torch.cuda.device_count() or 1,
        workers_per_device=4
    )

    # server = ls.LitServer(
    #     AdsHandler(),
    #     accelerator="cpu",
    #     devices=1,
    #     workers_per_device=1
    # )
    server.run(host="0.0.0.0", port=5005)
