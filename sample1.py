from sklearn.externals import joblib  # You may need to use joblib or pickle for loading trained models
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np

# ----- Load Pre-Trained Models -----
# Assuming models are stored in 'knn_model.pkl', 'rf_model.pkl' and scaler is stored as 'scaler.pkl'
knn = joblib.load('knn_model.pkl')
rf = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# ----- Email Alert System -----
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

# ----- Real-Time Detection -----
def detect_and_alert_new_input(new_input):
    # Scale the new input
    new_input_scaled = scaler.transform([new_input])  # Assumes new_input is a list of feature values

    # Predict using both models (KNN and Random Forest)
    knn_prediction = knn.predict(new_input_scaled)
    rf_prediction = rf.predict(new_input_scaled)

    # Check if either model flags the input as malicious (assuming 'malicious' label is 1)
    malicious_label = 1  # Adjust based on your dataset's encoding for malicious activity
    if knn_prediction[0] == malicious_label or rf_prediction[0] == malicious_label:
        print("Unauthorized access detected! Sending alert...")
        send_alert_email(
            subject="Unauthorized Access Detected!",
            body=f"A potentially malicious access attempt was detected.\nInput data: {new_input}",
            to_email="admin@example.com"  # Replace with the administrator's email
        )
    else:
        print("Access appears to be normal.")

# ----- Simulate Real-Time Input -----
# Example input for testing (replace with actual data)
new_input_example = [0.1, 0.5, 0.3, 0.4, 0.6, 0.2, 0.8, 0.7, 0.9]  # Example feature values
detect_and_alert_new_input(new_input_example)
