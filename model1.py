import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import joblib

# Read data from CSV
def read_data_from_csv(csv_file):
    documents = []
    labels = []
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            documents.append(row[0])
            labels.append(row[1])
    return documents, labels

# Vectorize the documents
def vectorize_documents(documents):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents)
    return X, vectorizer

# Train the model
def train_model(X, labels):
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X, labels)
    return svm_classifier

# Save the trained model
def save_model(model, vectorizer, model_file, vectorizer_file):
    joblib.dump(model, model_file)
    joblib.dump(vectorizer, vectorizer_file)
    print("Model and vectorizer saved successfully.")

# Main function for training
def main():
    # Read data from CSV
    csv_file = r'D:\s8\Code\Analytics_of_Prescription-main\dataset.csv'
    
    
    documents, labels = read_data_from_csv(csv_file)

    # Vectorize the documents
    X, vectorizer = vectorize_documents(documents)

    # Train the model
    model = train_model(X, labels)

    # Save the trained model and vectorizer
    model_file = 'svm_model.pkl'
    vectorizer_file = 'vectorizer.pkl'
    save_model(model, vectorizer, model_file, vectorizer_file)

if __name__ == "__main__":
    main()
