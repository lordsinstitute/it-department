import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ====================== SAMPLE DATASET ======================
# 20 labeled examples (10 normal + 10 attacks). Expand later with real data.
data = {
    "text": [
        # Normal / Legitimate messages (label 0)
        "Hey, how's your day going? Want to grab lunch later?",
        "I saw your post about the new movie. It looks awesome!",
        "Reminder: Team meeting at 3 PM today.",
        "Can you share the notes from yesterday's lecture?",
        "Happy birthday! Wishing you all the best.",
        "What's the weather like there? It's raining here.",
        "Thanks for helping me with the assignment.",
        "Did you watch the game last night? Incredible finish!",
        "Let's plan a trip next month.",
        "Your profile picture is really nice!",
        # Social Engineering Attacks (label 1)
        "URGENT: Your account has been hacked! Verify immediately by clicking this link: http://secure-login-fake.com",
        "Congratulations! You've won $5000. Click here to claim your prize and provide your bank details.",
        "This is your bank. Suspicious activity detected. Please reply with your PIN to secure your account.",
        "Your friend is in emergency. Send $200 to this account number right now to help.",
        "Claim your free gift card! Limited time offer. Enter your email and phone number.",
        "Security alert from Meta: Your Facebook account will be suspended unless you log in here: fakefb.com/verify",
        "Invest now in this crypto scheme and get 300% returns in 24 hours. DM for details.",
        "Your package is on hold due to unpaid fee. Pay $49.99 here to release it: suspiciouslink.com",
        "We need your personal information to process your tax refund urgently.",
        "Click this link to update your password or lose access to your social media account.",
    ],
    "label": [0]*10 + [1]*10
}

df = pd.DataFrame(data)

# ====================== STREAMLIT APP ======================
st.set_page_config(page_title="Social Engineering Detector", page_icon="🔍", layout="wide")
st.title("🔍 AI-Based Social Engineering Attack Detection")
st.markdown("**Detects phishing, scams, and manipulation tactics on social media using Machine Learning (NLP)**")

st.sidebar.header("Project Details")
st.sidebar.info("Mini Project • TF-IDF + Logistic Regression")
st.sidebar.write("Dataset: 20 demo samples (expandable)")
st.sidebar.write("**How it works**: Text → TF-IDF vectors → Logistic Regression classifier")

# Show dataset
with st.expander("📊 View Sample Dataset"):
    st.dataframe(df)
    st.caption("Label 0 = Normal • Label 1 = Social Engineering Attack")

# Train model
st.subheader("🤖 Train the AI Model")
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.accuracy = None
    st.session_state.report = None

if st.button("🚀 Train Model", type="primary"):
    with st.spinner("Training..."):
        X = df['text']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        model_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=500)),
            ('classifier', LogisticRegression(max_iter=1000))
        ])

        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Normal', 'Attack'], output_dict=True)

        st.session_state.model = model_pipeline
        st.session_state.accuracy = acc
        st.session_state.report = report

        st.success(f"✅ Model trained! Accuracy: **{acc:.2f}** ({acc*100:.1f}%)")

        # Metrics
        st.subheader("📈 Performance Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{acc:.2f}")
        with col2:
            st.dataframe(pd.DataFrame(report).transpose().round(2))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Attack'],
                    yticklabels=['Normal', 'Attack'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

# Prediction
st.subheader("🔎 Real-Time Detection")
user_input = st.text_area("Paste any social media post / DM / comment here:", height=150,
                          placeholder="Example: URGENT: Your account is suspended...")

if st.button("Analyze Message", type="primary"):
    if st.session_state.model is None:
        st.error("Train the model first!")
    elif not user_input.strip():
        st.warning("Enter some text!")
    else:
        with st.spinner("Analyzing..."):
            prediction = st.session_state.model.predict([user_input])[0]
            probabilities = st.session_state.model.predict_proba([user_input])[0]
            confidence = max(probabilities) * 100

            if prediction == 1:
                st.error("🚨 POTENTIAL SOCIAL ENGINEERING ATTACK DETECTED!")
                st.write(f"**Confidence:** {confidence:.1f}%")
                st.info("Common red flags: urgency, suspicious links, requests for money/PIN/password.")
            else:
                st.success("✅ Looks like a NORMAL / SAFE message.")
                st.write(f"**Confidence:** {confidence:.1f}%")

st.caption("Mini Project • Expand the dataset for even better accuracy • Future: Add BERT / social media API integration")
