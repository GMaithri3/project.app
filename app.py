import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from wordcloud import WordCloud
import io
import numpy as np
from collections import Counter  # Added missing import for Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# App configuration - MUST be first Streamlit command
st.set_page_config(page_title='Sentiment Analysis Deployment', page_icon='📊', layout='wide')

# Download NLTK data (no decorator; call after config)
def load_nltk():
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
    return SentimentIntensityAnalyzer(), stopwords.words('english')

sia, stop_words = load_nltk()

# Load dataset (no decorator; call after config)
def load_dataset():
    try:
        df = pd.read_excel('dataset -P582.xlsx', sheet_name='Sheet1')
        df['text'] = df['title'].fillna('').astype(str) + ' ' + df['body'].fillna('').astype(str)
        return df
    except FileNotFoundError:
        st.error("Dataset file 'dataset -P582.xlsx' not found. Please upload it.")
        return None

df = load_dataset()

# Preprocessing function
def preprocess_text(text):
    if pd.isna(text):
        return ''
    text = re.sub(r'http\S+|www\S+|https\S+', '', str(text))
    text = re.sub(r'@\w+|#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text.lower().strip()

# VADER sentiment function
def get_vader_sentiment(text):
    processed = preprocess_text(text)
    scores = sia.polarity_scores(processed)
    compound = scores['compound']
    if compound >= 0.05:
        label = 'positive'
    elif compound <= -0.05:
        label = 'negative'
    else:
        label = 'neutral'
    return label, compound, scores

# True sentiment from rating
def get_true_sentiment(rating):
    if pd.isna(rating):
        return 'neutral'
    rating = int(rating)
    if rating <= 2:
        return 'negative'
    elif rating == 3:
        return 'neutral'
    else:
        return 'positive'

st.title('📊 NLP Sentiment Analysis Deployment: Customer Reviews')

# Sidebar
st.sidebar.header('Navigation')
page = st.sidebar.selectbox('Select Section:', ['Dashboard', 'EDA', 'Model Evaluation', 'ML Models', 'Predict New Review'])

# Analyze dataset if not done
if 'df_analyzed' not in st.session_state and df is not None:
    with st.spinner('Analyzing dataset, VADER sentiments, and training ML models...'):
        df['processed_text'] = df['text'].apply(preprocess_text)
        
        # Compute sentiments and scores in one go for efficiency
        def compute_sentiment(x):
            label, compound, scores = get_vader_sentiment(x)
            return pd.Series({
                'predicted_sentiment': label,
                'compound': compound,
                'vader_neg': scores['neg'],
                'vader_neu': scores['neu'],
                'vader_pos': scores['pos']
            })
        
        sentiment_df = df['processed_text'].apply(compute_sentiment)
        df = pd.concat([df, sentiment_df], axis=1)
        df['true_sentiment'] = df['rating'].apply(get_true_sentiment)
        
        # Drop any rows with NaN in key columns if needed
        df = df.dropna(subset=['processed_text', 'predicted_sentiment', 'true_sentiment'])
        
        # ML Models Preparation
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        inverse_label_map = {v: k for k, v in label_map.items()}
        y = df['true_sentiment'].map(label_map)
        X = df['processed_text']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Vectorizer
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Models
        models_dict = {
            'Naive Bayes': MultinomialNB(),
            'SVM': SVC(kernel='linear', probability=True),
            'Random Forest': RandomForestClassifier(n_estimators=100),
            'Decision Tree': DecisionTreeClassifier(max_depth=10),
            'KNN': KNeighborsClassifier(n_neighbors=5)
        }
        
        # Train and evaluate models
        trained_models = {}
        model_results = {}
        for name, model in models_dict.items():
            trained_model = model.fit(X_train_vec, y_train)
            y_pred = trained_model.predict(X_test_vec)
            acc = accuracy_score(y_test, y_pred)
            model_results[name] = {'model': trained_model, 'accuracy': acc, 'y_pred': y_pred}
            trained_models[name] = trained_model
        
        # Store in session state
        st.session_state.df_analyzed = df
        st.session_state.vectorizer = vectorizer
        st.session_state.trained_models = trained_models
        st.session_state.model_results = model_results
        st.session_state.y_test = y_test
        st.session_state.label_map = label_map
        st.session_state.inverse_label_map = inverse_label_map

if df is None or 'df_analyzed' not in st.session_state:
    st.stop()

df_analyzed = st.session_state.df_analyzed

if page == 'Dashboard':
    st.header('Overview Dashboard')
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_reviews = len(df_analyzed)
        st.metric('Total Reviews', total_reviews)
    with col2:
        avg_rating = df_analyzed['rating'].mean()
        st.metric('Average Rating', f"{avg_rating:.2f}")
    with col3:
        pos_pct = (df_analyzed['true_sentiment'] == 'positive').mean() * 100
        st.metric('Positive %', f"{pos_pct:.1f}%")
    with col4:
        acc = accuracy_score(df_analyzed['true_sentiment'], df_analyzed['predicted_sentiment'])
        st.metric('Model Accuracy', f"{acc:.4f}")
    
    col5, col6 = st.columns(2)
    with col5:
        fig_rating = px.histogram(df_analyzed, x='rating', nbins=5, title='Rating Distribution')
        st.plotly_chart(fig_rating, use_container_width=True)
    with col6:
        sentiment_counts = df_analyzed['true_sentiment'].value_counts()
        fig_sent = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title='True Sentiment Distribution')
        st.plotly_chart(fig_sent, use_container_width=True)
    
    st.subheader('Sample Analyzed Reviews')
    st.dataframe(df_analyzed[['title', 'rating', 'predicted_sentiment', 'compound']].head(10))

elif page == 'EDA':
    st.header('Exploratory Data Analysis')
    
    # Word Cloud
    st.subheader('Word Cloud from All Reviews')
    all_text = ' '.join(df_analyzed['processed_text'].dropna())
    if all_text:
        wc = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_text)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)
    
    # Top Words
    st.subheader('Top 20 Frequent Words')
    all_words = ' '.join(df_analyzed['processed_text']).split()
    word_counts = pd.DataFrame(Counter(all_words).most_common(20), columns=['Word', 'Count'])
    fig_words = px.bar(word_counts, x='Word', y='Count', title='Top Words')
    st.plotly_chart(fig_words, use_container_width=True)
    
    # Sentiment vs Rating Scatter
    st.subheader('Compound Score vs Rating')
    fig_scatter = px.scatter(df_analyzed, x='rating', y='compound', color='predicted_sentiment', title='VADER Compound vs True Rating')
    st.plotly_chart(fig_scatter, use_container_width=True)

elif page == 'Model Evaluation':
    st.header('VADER Model Evaluation')
    
    y_true = df_analyzed['true_sentiment']
    y_pred = df_analyzed['predicted_sentiment']
    
    st.metric('Overall Accuracy', f"{accuracy_score(y_true, y_pred):.4f}")
    
    st.subheader('Classification Report')
    st.text(classification_report(y_true, y_pred))
    
    # Confusion Matrix
    labels = ['negative', 'neutral', 'positive']
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig_cm, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title('Confusion Matrix')
    st.pyplot(fig_cm)
    
    # Normalized CM
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig_norm, ax_norm = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax_norm)
    ax_norm.set_title('Normalized Confusion Matrix')
    st.pyplot(fig_norm)
    
    # Distributions
    col1, col2 = st.columns(2)
    with col1:
        true_counts = y_true.value_counts()
        fig_true = px.pie(values=true_counts.values, names=true_counts.index, title='True Sentiments')
        st.plotly_chart(fig_true, use_container_width=True)
    with col2:
        pred_counts = y_pred.value_counts()
        fig_pred = px.pie(values=pred_counts.values, names=pred_counts.index, title='Predicted Sentiments')
        st.plotly_chart(fig_pred, use_container_width=True)

elif page == 'ML Models':
    st.header('5 Different ML Models Building and Evaluation')
    
    model_results = st.session_state.model_results
    
    # Comparison
    accuracies = pd.DataFrame([
        {'Model': name, 'Accuracy': res['accuracy']} 
        for name, res in model_results.items()
    ]).sort_values('Accuracy', ascending=False)
    
    st.subheader('Model Accuracies Comparison')
    fig = px.bar(accuracies, x='Model', y='Accuracy', color='Accuracy', title='Test Accuracies')
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(accuracies)
    
    # Detailed evaluation for selected model
    selected = st.selectbox('Select Model for Detailed Evaluation:', list(model_results.keys()))
    if selected:
        res = model_results[selected]
        y_pred_num = res['y_pred']
        y_test_num = st.session_state.y_test
        inverse_label_map = st.session_state.inverse_label_map
        y_test_lab = [inverse_label_map[i] for i in y_test_num]
        y_pred_lab = [inverse_label_map[i] for i in y_pred_num]
        
        st.subheader(f'{selected} - Classification Report')
        st.text(classification_report(y_test_lab, y_pred_lab))
        
        # Confusion Matrix
        labels = ['negative', 'neutral', 'positive']
        cm = confusion_matrix(y_test_lab, y_pred_lab, labels=labels)
        fig_cm, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(f'{selected} Confusion Matrix')
        st.pyplot(fig_cm)

elif page == 'Predict New Review':
    st.header('Predict Sentiment for New Review')
    
    with st.form('review_form'):
        title = st.text_input('Review Title:')
        body = st.text_area('Review Body:', height=150)
        submitted = st.form_submit_button('Analyze')
    
    if submitted:
        if title or body:
            full_text = title + ' ' + body
            processed = preprocess_text(full_text)
            show_wc = st.checkbox('Show Word Cloud for this review')
            
            if show_wc:
                if processed:
                    wc_single = WordCloud(width=400, height=200, background_color='white').generate(processed)
                    fig_single, ax_single = plt.subplots(figsize=(6, 3))
                    ax_single.imshow(wc_single, interpolation='bilinear')
                    ax_single.axis('off')
                    st.pyplot(fig_single)
            
            predictor = st.selectbox('Select Predictor:', ['VADER'] + list(st.session_state.trained_models.keys()))
            
            if predictor == 'VADER':
                label, compound, scores = get_vader_sentiment(full_text)
                st.success(f'**Predicted Sentiment:** {label.upper()}')
                st.info(f'**Compound Score:** {compound:.4f}')
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric('Positive Score', f"{scores['pos']:.4f}")
                with col2:
                    st.metric('Negative Score', f"{scores['neg']:.4f}")
                with col3:
                    st.metric('Neutral Score', f"{scores['neu']:.4f}")
            else:
                vec = st.session_state.vectorizer.transform([processed])
                pred_num = st.session_state.trained_models[predictor].predict(vec)[0]
                label = st.session_state.inverse_label_map[pred_num]
                st.success(f'**Predicted Sentiment:** {label.upper()}')
                
                # Probabilities if available
                try:
                    probs = st.session_state.trained_models[predictor].predict_proba(vec)[0]
                    max_proba = np.max(probs)
                    st.info(f'**Confidence:** {max_proba:.4f}')
                    
                    prob_df = pd.DataFrame({
                        'Sentiment': ['negative', 'neutral', 'positive'],
                        'Probability': probs
                    })
                    fig_prob = px.bar(prob_df, x='Sentiment', y='Probability', title='Sentiment Probabilities')
                    st.plotly_chart(fig_prob, use_container_width=True)
                except:
                    st.info('Probabilities not available for this model.')
        else:
            st.warning('Please enter a title or body.')

# Download button
st.sidebar.markdown('---')
csv = df_analyzed.to_csv(index=False).encode('utf-8')
st.sidebar.download_button('Download Analyzed Dataset', csv, 'analyzed_reviews.csv', 'text/csv')

# Footer
st.markdown('---')

st.markdown('**Project: NLP Sentiment Analysis using VADER and ML Models** | **Deployment: Streamlit App**')
