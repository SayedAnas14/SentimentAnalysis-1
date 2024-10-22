import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder


# Function to clean and preprocess text data
def clean_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Convert to lowercase
    text = text.lower()

    return text


# Load the dataset
df = pd.read_excel('products.xlsx')

# Drop duplicates and rows with missing values
df = df.drop_duplicates()
df = df.dropna(subset=['Review Text', 'Rating'])

# Clean the review text
df['Cleaned Review Text'] = df['Review Text'].apply(clean_text)

# Convert ratings to numerical values (assuming 1-5 scale)
df['Rating'] = df['Rating'].str.extract(r'(\d+)')[0].astype(int)

# Define sentiment labels based on ratings
df['Sentiment'] = df['Rating'].apply(lambda x: 'positive' if x > 3 else ('neutral' if x == 3 else 'negative'))

# Prepare data for machine learning
X = df['Cleaned Review Text']
y = df['Sentiment']

# Encode sentiment labels to numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Function to plot ROC curve
def plot_roc_curve(model, X_test, y_test, model_name):
    y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
    fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')


# Train and evaluate Support Vector Machine model
svm_model = SVC(probability=True)  # Enable probability estimates
svm_model.fit(X_train_tfidf, y_train)
y_pred_svm = svm_model.predict(X_test_tfidf)
print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))
print("SVM Accuracy Score:", accuracy_score(y_test, y_pred_svm))

# Train and evaluate Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)
print("Logistic Regression Classification Report:")
print(classification_report(y_test, y_pred_lr))
print("Logistic Regression Accuracy Score:", accuracy_score(y_test, y_pred_lr))

# Train and evaluate Decision Tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_tfidf, y_train)
y_pred_dt = dt_model.predict(X_test_tfidf)
print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt))
print("Decision Tree Accuracy Score:", accuracy_score(y_test, y_pred_dt))

# Save the models and vectorizer
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(lr_model, 'logistic_regression_model.pkl')
joblib.dump(dt_model, 'decision_tree_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Models and vectorizer saved.")

# Visualization of sentiment distribution including product name
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Sentiment', palette='Set2', hue='Product Name')
plt.title('Sentiment Distribution by Product')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Product Name')
plt.show()

# Word Cloud Visualization for Positive, Negative, and Neutral reviews
positive_reviews = ' '.join(df[df['Sentiment'] == 'positive']['Cleaned Review Text'])
negative_reviews = ' '.join(df[df['Sentiment'] == 'negative']['Cleaned Review Text'])
neutral_reviews = ' '.join(df[df['Sentiment'] == 'neutral']['Cleaned Review Text'])

# Create word clouds
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
wordcloud_positive = WordCloud(width=400, height=200, background_color='white').generate(positive_reviews)
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis('off')
plt.title('Positive Reviews Word Cloud')

plt.subplot(1, 3, 2)
wordcloud_negative = WordCloud(width=400, height=200, background_color='white').generate(negative_reviews)
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis('off')
plt.title('Negative Reviews Word Cloud')

plt.subplot(1, 3, 3)
wordcloud_neutral = WordCloud(width=400, height=200, background_color='white').generate(neutral_reviews)
plt.imshow(wordcloud_neutral, interpolation='bilinear')
plt.axis('off')
plt.title('Neutral Reviews Word Cloud')

plt.show()

# Plot ROC Curves for all models
plt.figure(figsize=(10, 8))
plot_roc_curve(svm_model, X_test_tfidf, y_test, "SVM")
plot_roc_curve(lr_model, X_test_tfidf, y_test, "Logistic Regression")
plot_roc_curve(dt_model, X_test_tfidf, y_test, "Decision Tree")

plt.title('ROC Curves for Different Models')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()
