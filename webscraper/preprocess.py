import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib



# Function to clean and preprocess text data
def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    return text

# Load the dataset
df = pd.read_excel('products.xlsx')

# Drop duplicates and missing values
df = df.drop_duplicates()
df = df.dropna(subset=['Review Text', 'Rating'])

# Clean the review text
df['Cleaned Review Text'] = df['Review Text'].apply(clean_text)

# Convert ratings to numerical values and define sentiment labels
df['Rating'] = df['Rating'].str.extract(r'(\d+)')[0].astype(int)
df['Sentiment'] = df['Rating'].apply(lambda x: 'positive' if x > 3 else ('neutral' if x == 3 else 'negative'))

# Prepare data for machine learning
X = df['Cleaned Review Text']
y = df['Sentiment']

# Encode sentiment labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Function to evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

# --- SVM Model ---
print("SVM Model:")
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)
evaluate_model(svm_model, X_test_tfidf, y_test)

# --- Logistic Regression Model ---
print("\nLogistic Regression Model:")
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_tfidf, y_train)
evaluate_model(logistic_model, X_test_tfidf, y_test)

# --- Neural Network Model ---
print("\nNeural Network Model:")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(X_sequences, padding='post')

# Split padded sequences into train and test
X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(X_padded, y_encoded, test_size=0.2, random_state=42)

# Build the Neural Network model
model_nn = Sequential()
model_nn.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=128, input_length=X_padded.shape[1]))
model_nn.add(LSTM(128))
model_nn.add(Dense(3, activation='softmax'))

# Compile the model
model_nn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model_nn.fit(X_train_nn, y_train_nn, epochs=5, batch_size=32, validation_split=0.1)

# Evaluate the Neural Network model
loss, accuracy = model_nn.evaluate(X_test_nn, y_test_nn)
print(f"\nNeural Network Accuracy: {accuracy * 100:.2f}%")

# Save the models and vectorizer
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(logistic_model, 'logistic_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
model_nn.save('neural_network_model.h5')

print("Models and vectorizer saved.")
