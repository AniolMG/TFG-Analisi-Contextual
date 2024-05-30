from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import os
import numpy as np
import re
from transformers import BertTokenizer, TrainingArguments, Trainer
from sklearn.preprocessing import MultiLabelBinarizer
import webScraping
import torch

from datasets import Dataset
import pickle

import wandb

import csv

from model_multilabel import BertForMultiLabelSequenceClassification

from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize an empty dictionary to store the URL-label mappings
url_to_labels = {}

# Open the CSV file and read in the data
with open('url_labels.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header row
    for i, row in enumerate(reader, start=2):  # start counter from 2 to account for header row
        try:
            url, main_label, secondary_label = row
            # Add the URL to the dictionary with a list containing the main and secondary labels
            url_to_labels[url] = [main_label]
            if secondary_label:
                url_to_labels[url].append(secondary_label)
        except ValueError:
            print(f"Error at line {i}: {row}")
            raise


# Get list of .txt files in the "WebLinks" folder
web_links_dir = "WebLinks"
web_contents_dir = "WebContents"

# Initialize lists to store texts and labels
texts = []
labels = []
urls = []

# Iterate through each URL and extract text from URLs
for url, label_list in url_to_labels.items():
    print("Processing url ", url, "\n")
    
    # Create a safe filename from the URL
    filename = ''.join([c if c.isalnum() or c in '._-~' else '-' for c in url]) + '.txt'
    main_label = label_list[0]  # Get the main label from the label list
    content_file_path = os.path.join(web_contents_dir, main_label, filename)
    
    if os.path.exists(content_file_path):
        # If the content file exists, read the text from the file
        with open(content_file_path, 'r') as content_file:
            text = content_file.read()
    else:
        # If the content file doesn't exist, scrape the webpage and save the text in a new file
        text = webScraping.extract_text_from_url(url)
        if text:
            # Split the text into words
            words = text.split()
            # Keep only the first 2048 words
            words = words[:2048]
            # Join the words back into a string
            text = ' '.join(words)
            with open(content_file_path, 'w') as content_file:
                content_file.write(text)
        else:
            print("No s'ha pogut obtenir el text de la pàgina")
            continue  # Skip to the next URL if the webpage couldn't be scraped

    text = re.sub(r'[^a-zA-Z0-9\sÀ-ÿ$€£"%Δ.,/\'\-<>?!()@\’\‘]', ' ', text)
    text = re.sub(r'\s+', ' ', text)  # Replace more than 1 space with 1 space
    truncated_text = ' '.join(text.split()[:10])  # Extract first 10 words
    print(truncated_text, "\n")
    texts.append(text)
    print(len(texts))
    labels.append(label_list)  # Get labels from the dictionary
    urls.append(url)
    print(url, " es de ", label_list)

# Start a run, tracking hyperparameters
wandb.init(
    project="BERT",
    config={
        "metric": 'accuracy',
        "epochs": 10, #10
        "batch_size": 4,
        "learning_rate": 1e-5
    }
)

# Set environment variables
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/lib/cuda"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# Load tokenizer and model
model_name = "bert-base-multilingual-cased" # Use this for multiple languages
#model_name = "bert-base-cased" # Use this for English only
tokenizer = BertTokenizer.from_pretrained(model_name)

# Initialize the MultiLabelBinarizer
mlb = MultiLabelBinarizer()
# Fit the MultiLabelBinarizer and transform the labels
labels_encoded = mlb.fit_transform(labels).astype(np.float32)
num_labels = labels_encoded.shape[1]  # Number of columns in the matrix is the number of unique labels

model = BertForMultiLabelSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Create a Dataset object
ds = Dataset.from_dict({
    "texts": texts,
    "labels": labels_encoded,
    "urls": urls
})

# Define a tokenize function
def tokenize(examples):
    return tokenizer(examples["texts"], truncation=True, padding='max_length', max_length=512)

# Tokenize the texts
ds = ds.map(tokenize, batched=True)

# Split the dataset into training and test sets
ds = ds.train_test_split(test_size=0.2, seed=42)

# Define the training arguments
args = TrainingArguments(
    output_dir='./results', 
    learning_rate=wandb.config.learning_rate,
    per_device_train_batch_size=wandb.config.batch_size,
    num_train_epochs=wandb.config.epochs,
    report_to="wandb",  # Enable logging to W&B
)

# Define the test_urls here (if not it would be necessary to create a custom Trainer class)
test_urls = ds["test"]["urls"]

# Define the compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Convert logits to a PyTorch tensor
    logits = torch.tensor(logits)
    # Apply sigmoid function
    sigmoid_logits = torch.sigmoid(logits)
    # Set a threshold value
    threshold = 0.5
    # Anything above the threshold will be a predicted label
    predictions = (sigmoid_logits > threshold).int()

    # Convert predictions tensor to numpy array for sklearn
    predictions = predictions.detach().cpu().numpy()

    # Calculate metrics
    f1_individual = f1_score(labels, predictions, average=None)
    accuracy = accuracy_score(labels, predictions)

    # Compute the confusion matrix for each class
    cm = multilabel_confusion_matrix(labels, predictions)
    folder_path = 'Confusion Matrixes'
    # For each class...
    for i, matrix in enumerate(cm):
        plt.figure(figsize=(10,7))
    
        # Use seaborn to create a heatmap of the confusion matrix
        sns.heatmap(matrix, annot=True, fmt='d')
    
        # Add labels to the plot
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for Class {mlb.classes_[i]}')
    
        # Save the plot as an image file
        plt.savefig(f'{folder_path}/confusion_matrix_{mlb.classes_[i]}.png')

    # Find the indices of the incorrect predictions
    incorrects = np.where(labels != predictions)

    # Print the URL, true and predicted labels for the incorrect predictions
    incorrect_urls = {}
    for i in incorrects[0]:
        url = test_urls[i]
        if url not in incorrect_urls:
            true_label_names = mlb.inverse_transform(np.array([labels[i]]))
            predicted_label_names = mlb.inverse_transform(np.array([predictions[i]]))
            incorrect_urls[url] = (true_label_names, predicted_label_names)
    
    for url, (true_label_names, predicted_label_names) in incorrect_urls.items():
        print(f"URL: {url}, True label: {true_label_names}, Predicted label: {predicted_label_names}")
    
    print()
    print("F1 Score for each label: ")
    for label, score in zip(mlb.classes_, f1_individual):
        print(f"Label {label}: {score}")
    print("Global accuracy: ", accuracy)

    # Calculate the number of instances where at least one label has been predicted correctly
    correct_predictions = np.sum(np.logical_and(labels, predictions), axis=1) > 0
    num_correct_predictions = np.sum(correct_predictions)

    # Calculate the percentage of instances where at least one label has been predicted correctly
    percentage_correct_predictions = num_correct_predictions / len(labels) * 100

    print(f"Percentage of instances where at least one label has been predicted correctly: {percentage_correct_predictions}%")
    
    return {
        "f1_individual": f1_individual,
        "accuracy": accuracy,
        "percentage_correct_predictions": percentage_correct_predictions
    }

# Create a Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

# Save the model
model.save_pretrained('bert_model2 multilabel LAST')

# Save the MultiLabelBinarizer
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(mlb, f)

wandb.finish()
