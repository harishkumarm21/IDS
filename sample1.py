import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ----- 1. Load and Preprocess Data -----
# Load dataset (replace 'path_to_dataset.csv' with the actual path)
df = pd.read_csv('path_to_dataset.csv')

# Handle missing values
df = df.dropna()

# Encode labels if necessary (replace 'label' with actual column name)
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])

# Split features and target labels
X = df.drop('label', axis=1)  # Features
y = df['label']  # Target labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ----- 2. KNN Model -----
# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Make predictions and evaluate
y_pred_knn = knn.predict(X_test_scaled)
print("KNN Classification Report:")
print(classification_report(y_test, y_pred_knn))
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

# ----- 3. Random Forest Model -----
# Train Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions and evaluate
y_pred_rf = rf.predict(X_test)
print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# ----- 4. Email Alert System -----
def send_alert_email(subject, body, to_email):
    from_email = "your_email@example.com"  # Replace with your email
    password = "your_password"  # Use environment variables for security

    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.example.com", 587) as server:  # Replace with your SMTP server details
            server.starttls()
            server.login(from_email, password)
            server.sendmail(from_email, to_email, msg.as_string())
        print("Alert email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Trigger alert if malicious activity is detected
malicious_label = 1  # Adjust based on your dataset's encoding for malicious activity
for index, prediction in enumerate(y_pred_rf):
    if prediction == malicious_label:
        send_alert_email(
            subject="Suspicious Activity Detected!",
            body=f"Suspicious activity detected at index {index}.",
            to_email="admin@example.com"  # Replace with the administrator's email
        )
