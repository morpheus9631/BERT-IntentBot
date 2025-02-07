import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import time
import torch
import joblib

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from typing import List, Tuple, Union

from pandas import DataFrame
from wordcloud import WordCloud

from utils import PathUtil

# -----------------------------------------------------------------------------

def data_preparation(data_pathfile: str):

    with open(data_pathfile, 'r', encoding="utf-8") as in_file:
        data = json.load(in_file)
        
    in_df = pd.DataFrame(data['intents'])
    
    dic = {"tag":[], "patterns":[], "responses":[]}
    for i in range(len(in_df)):
        ptrns = in_df[in_df.index == i]['patterns'].values[0]
        rspns = in_df[in_df.index == i]['responses'].values[0]
        tag   = in_df[in_df.index == i]['tag'].values[0]
        for j in range(len(ptrns)):
            dic['tag'].append(tag)
            dic['patterns'].append(ptrns[j])
            dic['responses'].append(rspns)
        
    out_df = pd.DataFrame.from_dict(dic)
    
    return out_df

# -----------------------------------------------------------------------------

def plot_word_cloud(in_df: DataFrame, out_path: str):

    all_patterns = ' '.join(in_df['patterns'])

    # Generate a word cloud image
    wordcloud = WordCloud(
            background_color='white', 
            max_words=100, 
            contour_width=3, 
            contour_color='steelblue')\
        .generate(all_patterns)

    # Display the generated image:
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Do not show axes to keep it clean
    plt.title('Word Cloud for Patterns')
    
    out_pathfile = os.path.join(out_path, 'word_cloud.jpg')
    plt.savefig(out_pathfile)

# -----------------------------------------------------------------------------

def plot_patterns_len_dist(in_df: DataFrame, out_path: str):
    
    in_df['pattern_length'] = in_df['patterns'].apply(len)
    data_df = in_df['pattern_length']
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data_df, bins=30, kde=True)
    plt.title('Distribution of Pattern Lengths')
    plt.xlabel('Length of Patterns')
    plt.ylabel('Frequency')
    
    out_pathfile = os.path.join(out_path, 'patterns_len_dist.jpg')
    plt.savefig(out_pathfile)

# -----------------------------------------------------------------------------

def plot_intents_dist(in_df: DataFrame, out_path: str):
    
    data_df = in_df
    order   = in_df['tag'].value_counts().index
    
    plt.figure(figsize=(22, 16))
    sns.countplot(y='tag', data=data_df, order=order)
    plt.title('Distribution of Intents')
    plt.xlabel('Number of Patterns')
    plt.ylabel('Intent')
    
    out_pathfile = os.path.join(out_path, 'intents_dist.jpg')
    plt.savefig(out_pathfile)

# -----------------------------------------------------------------------------

def plot_num_unique_responses_per_intent(in_df: DataFrame, out_path: str):
    
    df_responses = in_df.explode('responses')
    
    df_unique_responses = df_responses\
        .groupby('tag')['responses']\
        .nunique()\
        .reset_index(name='unique_responses')
    
    plt.figure(figsize=(22, 16))
    sns.barplot(x='unique_responses', 
                y='tag', 
                data=df_unique_responses\
                    .sort_values('unique_responses', ascending=False))
    
    plt.title('Number of Unique Responses per Intent')
    plt.xlabel('Number of Unique Responses')
    plt.ylabel('Intent')

    out_pathfile = os.path.join(out_path, 'num_unique_responses_per_intent.jpg')
    plt.savefig(out_pathfile)


# -----------------------------------------------------------------------------

def plot_response_len_dist(in_df: DataFrame, out_path: str):
    
    df_responses = in_df.explode('responses')
    
    df_responses['response_length'] = df_responses['responses'].apply(len)
    plt.figure(figsize=(12, 8))
    sns.histplot(df_responses['response_length'], bins=30, kde=True)
    plt.title('Distribution of Response Lengths')
    plt.xlabel('Length of Responses')
    plt.ylabel('Frequency')    

    out_pathfile = os.path.join(out_path, 'response_len_dist.jpg')
    plt.savefig(out_pathfile)

# -----------------------------------------------------------------------------

def save_label_encoder(
    label_encoder: LabelEncoder, 
    saved_model_path: str
):
    """
    Save the trained label encoder to a file.
    """
    encoder_path = os.path.join(saved_model_path, 'label_encoder.pkl')
    joblib.dump(label_encoder, encoder_path)

    print(f"\nLabel encoder saved to {encoder_path}")

# -----------------------------------------------------------------------------

def save_optimizer(optimizer: AdamW, saved_model_path: str):
    """
    Save the optimizer state.
    """
    optimizer_path = os.path.join(saved_model_path, 'optimizer.pth')
    torch.save(optimizer.state_dict(), optimizer_path)
    
    print(f"Optimizer state saved to {optimizer_path}")

# -----------------------------------------------------------------------------

def preprocess_text(s):
    s = re.sub('[^a-zA-Z\']', ' ', s)  # Keep only alphabets and apostrophes
    s = s.lower()  # Convert to lowercase
    s = s.split()  # Split into words
    s = " ".join(s)  # Rejoin words to ensure clean spacing
    return s

# -----------------------------------------------------------------------------

def print_unique_value(in_df: DataFrame, colname: str, isSorted: bool=True):

    unique_values = in_df[colname].unique()
    if isSorted:
        unique_values = sorted(unique_values)
    
    print(f"\nUnique values of '{colname}': ", )
    print(json.dumps(unique_values, ensure_ascii=False, indent=2))

# -----------------------------------------------------------------------------

def  load_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer

# -----------------------------------------------------------------------------

def tokenize_text(
    text: str, 
    tokenizer: BertTokenizer, 
    max_len: int
) -> torch.Tensor:

    encoded_dict = tokenizer.encode_plus(
        text,                       # Input text
        add_special_tokens=True,    # Add '[CLS]' and '[SEP]'
        max_length=max_len,         # Pad or truncate to max length
        padding="max_length",       # Pad to max length
        truncation=True,
        return_attention_mask=True, # Construct attn. masks
        return_tensors='pt'         # Return pytorch tensors
    )
    
    
    return  encoded_dict

# -----------------------------------------------------------------------------

def encode_texts(tokenizer: BertTokenizer, texts: list, max_len):
    
    input_ids = []
    attention_masks = []
    
    for text in texts:
        encoded_dict = tokenize_text(text, tokenizer, max_len)
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    
    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

# -----------------------------------------------------------------------------

def load_label_encoder(labels: Union[List[str], np.ndarray]) -> LabelEncoder:
    
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    num_labels = len(label_encoder.classes_)
    
    return label_encoder, num_labels

# -----------------------------------------------------------------------------

def prepare_data(
    in_df: DataFrame,
    tokenizer: BertTokenizer,
    label_encoder: LabelEncoder,
    max_len: int,
    train_ratio: float
):
    X = in_df['patterns']
    y = in_df['tag']
    
    # # Encoding labels
    y_encoded = label_encoder.transform(y)
    
    # Encode the patterns
    input_ids, attention_masks = encode_texts(tokenizer, X, max_len)
    labels = torch.tensor(y_encoded)
    
    # Splitting the dataset into training and validation
    dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, labels)
    train_size = int(train_ratio * len(dataset))
    valid_size = len(dataset) - train_size
    
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16)
    
    return train_dataloader, valid_dataloader

# -----------------------------------------------------------------------------

def  build_model(num_labels, lr: float=2e-5):
    
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', 
        num_labels=num_labels)
    
    optimizer = AdamW(
        model.parameters(), 
        lr=lr, 
        no_deprecation_warning=True)
    
    return model, optimizer
    
# -----------------------------------------------------------------------------

def train_model(
    in_df: DataFrame, 
    device: torch.device,
    tokenizer: BertTokenizer,
    model: BertForSequenceClassification,
    optimizer: AdamW,
    best_model_path,
    max_len: int=128, 
    epochs: int=10,
    patience: int=3,
    train_ratio: float=0.8
):
    print(f"\nStrain training ...")
    start_time = time.time()
    
    model.to(device)
    best_loss = float("inf")
    patience_count = 0

    for epoch in range(epochs):
        
        # Prepare data
        train_dataloader, valid_dataloader = prepare_data(
            in_df,
            tokenizer,
            label_encoder, 
            max_len,
            train_ratio
        )
        
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        epochs_width = len(str(epochs))

        # --- Train mode ---
        for batch in train_dataloader:
            
            b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
            model.zero_grad()        

            outputs = model(
                b_input_ids, 
                token_type_ids=None, 
                attention_mask=b_input_mask, 
                labels=b_labels)
            
            loss = outputs[0]
            total_train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            # Compute accuracy during training
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            correct_train += torch.sum(predictions == b_labels).item()
            total_train += b_labels.size(0)

        # Training stats after epoch
        avg_train_loss = total_train_loss / len(train_dataloader)            
        train_accuracy = correct_train / total_train
        
        # print(f"Epoch {epoch+1}, Average Training Loss: {avg_train_loss:.2f}")
        print("Epoch {:>{}}".format(epoch+1, epochs_width), end='')
        print(f" - accuracy: {train_accuracy:.4f}" , end='')
        print(f", loss: {avg_train_loss:.4f}", end='')
        
        # --- Validation mode ---
        model.eval()
        total_val_loss = 0
        correct_valid = 0
        total_valid = 0
        
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch in valid_dataloader:
                b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)

                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                logits = outputs.logits

                total_val_loss += loss.item()

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                labels = b_labels.cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels)
                
                # Compute accuracy during training
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                correct_valid += torch.sum(predictions == b_labels).item()
                total_valid += b_labels.size(0)
                
        avg_valid_loss = total_val_loss / len(valid_dataloader)
        valid_accuracy = correct_valid / total_valid

        print(f" - val_accuracy: {valid_accuracy:.4f}", end='')
        print(f", val_loss: {avg_valid_loss:.4f}", end='')
        
        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            patience_count = 0
            
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f", Best model saved")
        else:
            patience_count += 1
            print()
            
        if patience_count >= patience:
            print()
            print(f"\nEarly stopping triggered after {epoch+1} epochs!")
            break 
            

    exec_time = time.time() - start_time
    print(f"\nTraining time: {exec_time:.2f} secs")

    return optimizer

# -----------------------------------------------------------------------------

def predict_intent(
    text: str, 
    model: BertForSequenceClassification, 
    tokenizer: BertTokenizer, 
    max_len
) -> Tuple:

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

    # Ptahs
    root_path = PathUtil.get_root_path()
    datasets_path = os.path.join(root_path, 'datasets')
    saved_model_path = os.path.join(root_path, 'saved_model')
    
    # model version
    version = str(int(time.time()))
    model_dir = f"model_{version}"
    best_model_path = os.path.join(saved_model_path, model_dir)
    PathUtil.ensure_path_exists(best_model_path)

    # Prepare data
    data_pathfile = os.path.join(datasets_path, 'intents.json')
    df_data = data_preparation(data_pathfile)

    # Preprocessing the dataset
    df_data['patterns'] = df_data['patterns'].apply(preprocess_text)
    df_data['tag'] = df_data['tag'].apply(preprocess_text)
    # print_unique_value(df_data, 'tag')
    
    print(f"\nInitial data count: {df_data.count()}")
    print(df_data.head(3))
    
    # Exploratory Data Analysis
    plot_path = os.path.join(root_path, 'outputs')
    PathUtil.ensure_path_exists(plot_path)
    
    # Exploratory Data Analysis
    plot_word_cloud(df_data, plot_path)
    plot_patterns_len_dist(df_data, plot_path)
    plot_intents_dist(df_data, plot_path)
    plot_num_unique_responses_per_intent(df_data, plot_path)
    plot_response_len_dist(df_data, plot_path)
    
    # Load tokenizer, label_encoder
    tokenizer = load_tokenizer()
    
    label_list = df_data['tag'].astype(str).tolist()
    label_encoder, num_labels = load_label_encoder(label_list)
    save_label_encoder(label_encoder, best_model_path)
    
    # Build and save best model
    model, optimizer = build_model(num_labels)
    
    # Train model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs  = 100
    max_len = 256
    patience = 5
    train_ratio = 0.8
    
    optimizer = train_model(
        in_df=df_data, 
        device=device, 
        tokenizer=tokenizer, 
        model=model, 
        optimizer=optimizer,        
        best_model_path=best_model_path,
        max_len=max_len,
        patience=patience,
        train_ratio=train_ratio,
        epochs=epochs)
    
    save_optimizer(optimizer, best_model_path)
    
    # Predict intent
    text = "I feel anxious today"
    intent, prob = predict_intent(text, model, tokenizer, max_len)
    print(f"Predicted Intent: {intent}")
    print(f"Probability: {prob:.4f}")
    
    