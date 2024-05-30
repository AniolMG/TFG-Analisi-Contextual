from transformers import BertTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import pickle
import webScraping
import re
import os
import requests

from model_multilabel import BertForMultiLabelSequenceClassification


# Load the trained model
model = BertForMultiLabelSequenceClassification.from_pretrained('_bert_model_multi_final') # Write the name of the folder containing the model here

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Load the MultiLabelBinarizer
with open('label_encoder.pkl', 'rb') as f:
    mlb = pickle.load(f)

# Get the list of label names
label_names = mlb.classes_.tolist()

# Tokenize new text
url = input("Entra un enllaç: ")
print("Processant enllaç: ", url, "\n")

# Load the URLs from the file
with open('Xarxes Socials.txt', 'r') as f:
    xarxes_socials_urls = f.read().splitlines()

# Check if the input URL matches any URL in the file
if any(xarxes_socials_url in url for xarxes_socials_url in xarxes_socials_urls):
    try:
        response = requests.get(url, timeout=15)
        # If the response was successful, no Exception will be raised
        response.raise_for_status()
    except:
        print('No es pot accedir a la pàgina.')
    else:
        print("Classificació determinada: Xarxes Socials")
else:
    text = webScraping.extract_text_from_url(url)
    if(text == None): 
        print("No s'ha pogut obtenir text.")
    
    else:   
        text = re.sub(r'[^a-zA-Z0-9\sÀ-ÿ$€£"%Δ.,/\'\-<>?!()@\’\‘]', ' ', text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        
        inputs = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        
        # Make prediction
        outputs = model(**inputs)
        
        # The model returns a tuple. The actual predictions are the first item in the tuple.
        predictions = outputs[0]
        
        # Multi Label behavior
        # Apply sigmoid function to get "probabilities" / confidence values
        probabilities = torch.sigmoid(predictions[0])
        
        # Get the values and indices of the top predictions
        top_probs, top_indices = torch.topk(probabilities, 6)
        
        # Get the names of the top predicted labels
        top_labels = [label_names[index] for index in top_indices]

        # For debugging or seeing all the results
        # print(f"Top predicted labels: {top_labels}")
        # print(f"Top predicted probabilities: {top_probs.tolist()}")
    
        # Filter labels with probability >= 0.5
        predicted_labels = [label for label, prob in zip(top_labels, top_probs) if prob >= 0.5]
        
        # Convert the list to a string and print without brackets
        predicted_category = '; '.join(predicted_labels)
        print()
        print(f"Classificació determinada: {predicted_category}")
        filtered_probs = [prob.item() for prob in top_probs if prob >= 0.5]  # Convert tensors to floats and filter
        print(f"Confiança de cada categoria: {filtered_probs}") # Where filtered probs contains top_probs.tolist() but filtered
