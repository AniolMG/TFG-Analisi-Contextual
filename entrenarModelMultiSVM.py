import os
import re
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import webScraping
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, multilabel_confusion_matrix
from sklearn.pipeline import make_pipeline
import csv

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

web_links_dir = "WebLinks"
web_contents_dir = "WebContents"

# Initialize lists to store texts, labels and URLs
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
    labels.append(label_list)  # Get labels from the dictionary
    urls.append(url)
    print(url, " es de ", label_list)

# Initialize the MultiLabelBinarizer
mlb = MultiLabelBinarizer()
# Fit the MultiLabelBinarizer and transform the labels
labels_encoded = mlb.fit_transform(labels).astype(np.float32)

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=20000)
X = vectorizer.fit_transform(texts)

# Split the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, labels_encoded, test_size=0.2, random_state=42)

# Initialize the base classifier
svc = SVC(kernel='linear', probability=True)

# Use OneVsRestClassifier for multilabel classification
classifier = OneVsRestClassifier(svc)

print("Begin training...")
# Train the classifier
classifier.fit(X_train, Y_train)
print("Training completed!")
# Make predictions
Y_pred = classifier.predict(X_test)

# Define the test URLs (for analysis purposes)
test_urls = [urls[i] for i in range(len(texts)) if i in X_test.indices]

# Compute metrics
def compute_metrics(Y_test, Y_pred):
    # Calculate metrics
    f1_individual = f1_score(Y_test, Y_pred, average=None)
    accuracy = accuracy_score(Y_test, Y_pred)

    # Compute the confusion matrix for each class
    cm = multilabel_confusion_matrix(Y_test, Y_pred)
    folder_path = 'Confusion Matrixes'
    # For each class...
    for i, matrix in enumerate(cm):
        plt.figure(figsize=(10,7))
        sns.heatmap(matrix, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for Class {mlb.classes_[i]}')
        plt.savefig(f'{folder_path}/confusion_matrix_{mlb.classes_[i]}.png')

    # Find the indices of the incorrect predictions
    incorrects = np.where(Y_test != Y_pred)

    # Print the URL, true and predicted labels for the incorrect predictions
    incorrect_urls = {}
    for i in incorrects[0]:
        url = test_urls[i]
        if url not in incorrect_urls:
            true_label_names = mlb.inverse_transform(np.array([Y_test[i]]))
            predicted_label_names = mlb.inverse_transform(np.array([Y_pred[i]]))
            incorrect_urls[url] = (true_label_names, predicted_label_names)
    
    for url, (true_label_names, predicted_label_names) in incorrect_urls.items():
        print(f"URL: {url}, True label: {true_label_names}, Predicted label: {predicted_label_names}")
    
    print()
    print("F1 Score for each label: ")
    for label, score in zip(mlb.classes_, f1_individual):
        print(f"Label {label}: {score}")
    print("Global accuracy: ", accuracy)

    # Calculate the number of instances where at least one label has been predicted correctly
    correct_predictions = np.sum(np.logical_and(Y_test, Y_pred), axis=1) > 0
    num_correct_predictions = np.sum(correct_predictions)

    # Calculate the percentage of instances where at least one label has been predicted correctly
    percentage_correct_predictions = num_correct_predictions / len(Y_test) * 100

    print(f"Percentage of instances where at least one label has been predicted correctly: {percentage_correct_predictions}%")
    
    return {
        "f1_individual": f1_individual,
        "accuracy": accuracy,
        "percentage_correct_predictions": percentage_correct_predictions
    }

# Compute and print metrics
metrics = compute_metrics(Y_test, Y_pred)
print(metrics)

# Save the model
with open('svm_model2.pkl', 'wb') as f:
    pickle.dump(classifier, f)

# Save the MultiLabelBinarizer
with open('label_encoder_SVM.pkl', 'wb') as f:
    pickle.dump(mlb, f)
