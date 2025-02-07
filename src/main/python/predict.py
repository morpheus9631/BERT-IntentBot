import joblib
import numpy as np
import os
import torch

from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification
from typing import Tuple

from utils import PathUtil


# -----------------------------------------------------------------------------

def load_model(model_path: str):
    """
    Load the trained model
    """
    tokenizer = BertTokenizer.from_pretrained(model_path)
    loaded_model = BertForSequenceClassification.from_pretrained(model_path)
    
    encoder_path = os.path.join(model_path, 'label_encoder.pkl')
    label_encoder = joblib.load(encoder_path)
    
    return loaded_model, tokenizer, label_encoder

# -----------------------------------------------------------------------------

def tokenize_text(text: str, tokenizer: BertTokenizer, max_len: int):
    """
    Tokenize the text and pad it to the maximum length
    """
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,  # Add [CLS] and [SEP]
        max_length=max_len,
        padding='max_length',    # Pad to max length
        truncation=True,         # Truncate long texts
        return_tensors='pt',     # Return in PyTorch tensors format
        return_attention_mask=True
    )
    return encoded_dict

# -----------------------------------------------------------------------------

def predict_intent(
    text: str, 
    model: BertForSequenceClassification, 
    tokenizer: BertTokenizer, 
    max_len: int=128
) -> Tuple:
    """
    Predict the intent of the text
    """
    # Tokenize and encode the text for BERT
    encoded_dict = tokenize_text(text, tokenizer, max_len)
    
    # Extract input IDs and attention masks from the encoded representation
    inputs = {key: value.to(device) for key, value in encoded_dict.items()}
    
    # No gradient calculation needed
    with torch.no_grad():
        model.eval()
        outputs = model(**inputs)
        logits = outputs.logits  # predicted unnormalized scores

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()

    # Use softmax to calculate probabilities
    probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()    
    
    # Get the predicted label with the highest probability
    predicted_label_idx = np.argmax(probabilities, axis=1).flatten()
    
    # Decode the predicted label
    predicted_label = label_encoder.inverse_transform(predicted_label_idx)[0]
    predicted_prob  = float(probabilities[0][predicted_label_idx].item())
    
    return predicted_label, predicted_prob

# -----------------------------------------------------------------------------


if __name__ == "__main__":
    
    root_path = PathUtil.get_root_path()
    saved_model_path = os.path.join(root_path, 'saved_model')
    
    # Load model    
    version = '1738943773'
    model_dir = f"model_{version}"
    best_model_path = os.path.join(saved_model_path, model_dir)
    
    # 定義設備為 CUDA 或 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入模型
    loaded_model, tokenizer, label_encoder = load_model(best_model_path)
    loaded_model.to(device)

    # 預測範例
    text = "I feel anxious today"
    intent, prob = predict_intent(text, loaded_model, tokenizer)
    print(f"Predicted Intent: {intent}")
    print(f"Probability: {prob:.4f}")
