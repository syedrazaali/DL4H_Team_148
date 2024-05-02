from gensim.scripts.glove2word2vec import glove2word2vec

glove_input_file = 'glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.word2vec.txt'
glove2word2vec(glove_input_file, word2vec_output_file)

from google.colab import drive
drive.mount('/content/drive')

# import  packages you need
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from google.colab import drive


data_dir = '/content/drive/My Drive/Deep Learning Data/'

def load_csv(file_name):
    # Function to load CSV and lower case column names
    df = pd.read_csv(f'{data_dir}{file_name}', low_memory=False)
    df.columns = df.columns.str.lower()  # Convert columns to lowercase
    return df

def load_data():
    # Loading all the necessary datasets
    datasets = {
        # 'admissions': load_csv('admissions_filtered.csv'),
        # 'chartevents': load_csv('filtered_CHARTEVENTS.csv'),
        # 'labevents': load_csv('filtered_LABEVENTS.csv'),
        # 'patients': load_csv('patients_filtered.csv'),
        # 'icustays': load_csv('icustays_filtered.csv'),
        # 'procedures_icd': load_csv('procedures_icd_full.csv'),
        # 'diagnoses_icd': load_csv('diagnoses_icd_full.csv'),
        'merged_data': load_csv('merged_data.csv')  # Loading the merged dataset
    }
    return datasets

def calculate_stats(df):
    print("\nData Statistics:")
    print(f"Total rows: {df.shape[0]}")
    print(f"Total columns: {df.shape[1]}")
    print(f"Columns: {df.columns.tolist()}")
    try:
        print(df.describe())  # Simplified to ensure compatibility across different pandas versions
        print(df.info())  # Provides data type for each column
    except Exception as e:
        print("Error in describing the data:", e)

# Load all datasets
datasets = load_data()

# Calculate statistics for all datasets to ensure consistency and understand the data
for name, dataset in datasets.items():
    print(f"{name.upper()} Data:")
    calculate_stats(dataset)

def missing_data_percentage(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data

# Load the merged data
merged_data = datasets['merged_data']
print(missing_data_percentage(merged_data))

# Handling missing values more contextually
merged_data['deathtime'].fillna('Not Applicable', inplace=True)  # Appropriate for non-existence of a death event

# I've assumed ICU-related missing data means no ICU stay. Fill time with admittime for continuity
merged_data['icu_los'].fillna(0, inplace=True)  # Zero length for no ICU stay
merged_data['icu_outtime'].fillna(merged_data['admittime'], inplace=True)  # Assuming no ICU stay ends at admission time
merged_data['icu_intime'].fillna(merged_data['admittime'], inplace=True)  # Assuming no ICU stay starts at admission time
merged_data['icustay_id'].fillna(-1, inplace=True)  # Use -1 as a placeholder for 'No ICU stay'

# Diagnosis missing values are filled with 'Unknown'
merged_data['diagnosis'].fillna('Unknown', inplace=True)

# Convert date columns to datetime and handle potential errors
date_columns = ['admittime', 'dischtime', 'icu_intime', 'icu_outtime']
for col in date_columns:
    merged_data[col] = pd.to_datetime(merged_data[col], errors='coerce')

# Calculating lengths only after date imputations to avoid negative or zero values unexpectedly
merged_data['hospital_stay_length'] = (merged_data['dischtime'] - merged_data['admittime']).dt.days.clip(lower=0)
merged_data['icu_stay_length'] = (merged_data['icu_outtime'] - merged_data['icu_intime']).dt.total_seconds() / 86400
merged_data['icu_stay_length'] = merged_data['icu_stay_length'].clip(lower=0)

# Calculate time from admission to ICU
merged_data['time_to_icu'] = (merged_data['icu_intime'] - merged_data['admittime']).dt.total_seconds() / 3600
merged_data['time_to_icu'] = merged_data['time_to_icu'].clip(lower=0)

# Check if any NaN values remain and print the updated stats
print(merged_data.isnull().sum())

# Check and correct gender inconsistencies
print("Unique gender values before:", merged_data['gender'].unique())
merged_data['gender_male'] = (merged_data['gender'] == 'M').astype(int)
print("Unique gender values after encoding:", merged_data['gender_male'].unique())

# Recalculate hospital and ICU stay lengths to correct potential errors
merged_data['hospital_stay_length'] = (merged_data['dischtime'] - merged_data['admittime']).dt.days
merged_data['icu_stay_length'] = (merged_data['icu_outtime'] - merged_data['icu_intime']).dt.total_seconds() / 86400
merged_data['hospital_stay_length'] = merged_data['hospital_stay_length'].clip(lower=0)
merged_data['icu_stay_length'] = merged_data['icu_stay_length'].clip(lower=0)

# Create features
merged_data['age_times_icu_length'] = merged_data['age_at_admission'] * merged_data['icu_stay_length']

# Check new statistics after cleaning and feature engineering
print(merged_data.describe())

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Stopwords and lemmatizer initialization
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Remove punctuations and numbers
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenization
    tokens = text.split()
    # Removing stopwords and lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Apply text preprocessing on the 'all_notes' column
merged_data['all_notes'] = merged_data['all_notes'].fillna('not available').apply(preprocess_text)

# Example to check the preprocessing output
print(merged_data['all_notes'].head())

# Save the processed data to a CSV file
data_dir = '/content/drive/My Drive/Deep Learning Data/'  # Specify your data directory
merged_data.to_csv(data_dir + 'merged_data_processed.csv', index=False)
merged_data = pd.read_csv(data_dir + 'merged_data_processed.csv')

glove_input_file = data_dir + 'glove.6B.100d.txt'  # Update the file name if needed

# Load the GloVe model directly without converting to word2vec format
model = KeyedVectors.load_word2vec_format(glove_input_file, binary=False, no_header=True)

def vectorize_note(note, embedding_model):
    # Ensure the note is treated as a string
    note = str(note)
    # Tokenize the note and filter tokens that are in the embedding model
    vectors = [embedding_model[word] for word in note.split() if word in embedding_model.key_to_index]  # use embedding_model.key_to_index to check words
    # Calculate the mean vector for the note, or return a zero vector if no words matched
    return np.mean(vectors, axis=0) if vectors else np.zeros(embedding_model.vector_size)

# Assuming merged_data['all_notes'] exists and contains the text data
note_vectors = [vectorize_note(note, model) for note in merged_data['all_notes']]
note_vectors_array = np.array(note_vectors)

# Print the shape of the resulting vectors to ensure they are correct
print("Shape of note_vectors_array:", note_vectors_array.shape)

# Save the array of vectors to a binary file in NumPy `.npy` format
np.save(data_dir + 'note_vectors_array.npy', note_vectors_array)

# load this array directly without reprocessing the text later to save time
note_vectors_array = np.load(data_dir + 'note_vectors_array.npy')
# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Histograms
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(merged_data['hospital_stay_length'], bins=30, ax=axes[0], kde=True)
axes[0].set_title('Histogram of Hospital Stay Length')
sns.histplot(merged_data['icu_stay_length'], bins=30, ax=axes[1], kde=True)
axes[1].set_title('Histogram of ICU Stay Length')
sns.histplot(merged_data['age_at_admission'], bins=30, ax=axes[2], kde=True)
axes[2].set_title('Histogram of Age at Admission')

# Box plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.boxplot(x=merged_data['hospital_stay_length'], ax=axes[0])
axes[0].set_title('Box Plot of Hospital Stay Length')
sns.boxplot(x=merged_data['icu_stay_length'], ax=axes[1])
axes[1].set_title('Box Plot of ICU Stay Length')
sns.boxplot(x=merged_data['age_at_admission'], ax=axes[2])
axes[2].set_title('Box Plot of Age at Admission')

# Bar chart for gender
plt.figure(figsize=(6, 4))
sns.countplot(x='gender', data=merged_data)
plt.title('Gender Distribution')

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(merged_data[['hospital_stay_length', 'icu_stay_length', 'age_at_admission', 'mortality_label']].corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')

plt.show()

# Convert each entry to a string and then calculate the length in words
merged_data['note_length'] = merged_data['all_notes'].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(10, 5))
sns.histplot(merged_data['note_length'], bins=50, kde=True)
plt.title('Distribution of Note Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')

# Common word analysis
from collections import Counter
# Ensure all entries are treated as strings
all_words = Counter(" ".join(merged_data['all_notes'].astype(str)).split())
most_common_words = all_words.most_common(20)
words, counts = zip(*most_common_words)
plt.figure(figsize=(10, 5))
sns.barplot(x=list(words), y=list(counts))
plt.title('Most Common Words in Clinical Notes')
plt.xticks(rotation=45)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.show()

# Assuming 'merged_data' and 'note_vectors_array.npy' are loaded
note_vectors_array = np.load(data_dir + 'note_vectors_array.npy')

# Print the shape of the note_vectors_array
print("Shape of note_vectors_array:", note_vectors_array.shape)

# Define feature columns explicitly based on the previous descriptions and outputs
feature_columns = [
    'age_at_admission', 'icu_los', 'hospital_stay_length', 'time_to_icu',
    'age_times_icu_length', 'gender_male'  # 'gender_male' added as it's created during preprocessing
]

# Prepare the numerical data
numerical_features = merged_data[feature_columns].values.astype(np.float32)

# Print the shape of the numerical_features
print("Shape of numerical_features:", numerical_features.shape)

# Now attempt to combine the numerical features and the note vectors
try:
    combined_features = np.hstack((numerical_features, note_vectors_array))
except ValueError as e:
    print("Caught ValueError:", e)
    # Additional prints that might help understand the mismatch
    print("First few entries of numerical_features:\n", numerical_features[:5])
    print("First few entries of note_vectors_array:\n", note_vectors_array[:5])

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split

# Assuming 'merged_data' and 'note_vectors_array.npy' are loaded
note_vectors_array = np.load(data_dir + 'note_vectors_array.npy')

# Define feature columns explicitly based on the previous descriptions and outputs
feature_columns = [
    'age_at_admission', 'icu_los', 'hospital_stay_length', 'time_to_icu',
    'age_times_icu_length', 'gender_male'  # 'gender_male' added as it's created during preprocessing
]

# Prepare the numerical data
numerical_features = merged_data[feature_columns].values.astype(np.float32)

# Combine the numerical features and the note vectors
combined_features = np.hstack((numerical_features, note_vectors_array))

# Convert features and targets into torch tensors
features_tensor = torch.tensor(combined_features, dtype=torch.float32)
targets_tensor = torch.tensor(merged_data['mortality_label'].values, dtype=torch.float32).view(-1, 1)

# Split data into train and test sets
features_train, features_test, targets_train, targets_test = train_test_split(
    features_tensor, targets_tensor, test_size=0.2, random_state=42
)

# Dataset and DataLoader setup for both train and test sets
train_dataset = TensorDataset(features_train, targets_train)
test_dataset = TensorDataset(features_test, targets_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the model
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Adjust for bidirectional

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# Setting up the model, loss function, and optimizer
input_size = combined_features.shape[1]
model = MyModel(input_size, 64, 1)
loss_func = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # lowered learning rate

# Scheduler setup
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# Metrics storage
epoch_accuracies = []
epoch_precisions = []
epoch_recalls = []
epoch_f1s = []
epoch_aucs = []
train_losses = []

# Training function for one epoch
def train_model_one_epoch(model, train_loader, loss_func, optimizer):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Execute the training loop
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train_model_one_epoch(model, train_loader, loss_func, optimizer)
    train_losses.append(train_loss)
    val_loss = evaluate_model(model, test_loader)[0]  # assume [0] is the loss, adjust if different
    scheduler.step(val_loss)  # Update the learning rate based on the validation loss
    accuracy, precision, recall, f1, auc = evaluate_model(model, test_loader)
    epoch_accuracies.append(accuracy)
    epoch_precisions.append(precision)
    epoch_recalls.append(recall)
    epoch_f1s.append(f1)
    epoch_aucs.append(auc)
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}, AUC: {auc:.2f}")

# Function to evaluate the model, RUN THIS BEFORE THE ABOVE CELL AS THIS FUNCTION IS CALLED WITHIN THE TRAINING LOOP
def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    predictions, actuals = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            predicted_classes = (torch.sigmoid(outputs) > 0.5).int()  # Convert probabilities to binary output
            predictions.extend(predicted_classes.view(-1).cpu())
            actuals.extend(targets.view(-1).cpu())

    predictions = [p.item() for p in predictions]
    actuals = [a.item() for a in actuals]

    # Calculate metrics
    accuracy = accuracy_score(actuals, predictions)
    precision = precision_score(actuals, predictions)
    recall = recall_score(actuals, predictions)
    f1 = f1_score(actuals, predictions)
    auc = roc_auc_score(actuals, predictions)

    return accuracy, precision, recall, f1, auc

# Run evaluation
accuracy, precision, recall, f1, auc = evaluate_model(model, test_loader)
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"AUC: {auc:.2f}")

# Visualize metrics
plt.figure(figsize=(15, 7))
plt.subplot(2, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(2, 3, 2)
plt.plot(epoch_accuracies, label='Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot(2, 3, 3)
plt.plot(epoch_precisions, label='Precision')
plt.title('Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')

plt.subplot(2, 3, 4)
plt.plot(epoch_recalls, label='Recall')
plt.title('Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')

plt.subplot(2, 3, 5)
plt.plot(epoch_f1s, label='F1 Score')
plt.title('F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')

plt.subplot(2, 3, 6)
plt.plot(epoch_aucs, label='AUC')
plt.title('AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')

plt.legend()
plt.tight_layout()
plt.show()
