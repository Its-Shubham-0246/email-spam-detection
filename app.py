import streamlit as st
import pandas as pd
import re
import nltk
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()
DATA_PATH = Path('data/spam.csv')

MODEL_FACTORY = {
    'Random Forest': lambda: RandomForestClassifier(n_estimators=200, random_state=42),
    'Decision Tree': lambda: DecisionTreeClassifier(random_state=42),
    'Multinomial NB': lambda: MultinomialNB()
}

TEXT_COLUMN_CANDIDATES = [
    'message', 'text', 'body', 'content', 'email', 'subject', 'review', 'sms', 'description', 'comment'
]
LABEL_COLUMN_CANDIDATES = [
    'label', 'target', 'class', 'category', 'spam', 'is_spam', 'y', 'result'
]

PAGE_STYLE = """
<style>
.bold-title { font-size: 2.8rem; font-weight: 800; margin-bottom: 0; }
.subtitle { color: #475569; margin-top: 0.1rem; margin-bottom: 1.2rem; }
.section-card { background: #ffffff; border-radius: 18px; padding: 28px; box-shadow: 0 16px 50px rgba(15, 23, 42, 0.08); margin-bottom: 24px; }
.sidebar-card { background: #f8fafc; border-radius: 16px; padding: 18px; }
</style>
"""


def infer_text_column(df):
    lower_cols = {c.lower(): c for c in df.columns}
    for name in TEXT_COLUMN_CANDIDATES:
        if name in lower_cols:
            return lower_cols[name]

    best_col = None
    best_score = -1
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object:
            sample = df[c].dropna().astype(str).head(100)
            avg_len = sample.map(len).mean() if len(sample) > 0 else 0
            if avg_len > best_score:
                best_score = avg_len
                best_col = c
    return best_col or df.columns[0]


def infer_label_column(df):
    lower_cols = {c.lower(): c for c in df.columns}
    for name in LABEL_COLUMN_CANDIDATES:
        if name in lower_cols:
            return lower_cols[name]

    for c in df.columns:
        vals = df[c].dropna().astype(str).str.lower().unique()
        if len(vals) <= 3 and set(vals).issubset({'spam', 'ham', '0', '1', 'true', 'false', 'yes', 'no'}):
            return c
    return None


def normalize_label_values(series):
    values = series.astype(str).str.strip().str.lower()
    if set(values.unique()).issubset({'0', '1'}):
        return values.map({'0': 'ham', '1': 'spam'})
    if set(values.unique()).issubset({'true', 'false'}):
        return values.map({'false': 'ham', 'true': 'spam'})
    if 'spam' in values.unique() or 'ham' in values.unique():
        return values.map({'spam': 'spam', 'ham': 'ham'})
    if 'yes' in values.unique() or 'no' in values.unique():
        return values.map({'no': 'ham', 'yes': 'spam'})
    return values


def load_data(uploaded_file=None):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='ISO-8859-1', low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding='utf-8', low_memory=False)
        source = getattr(uploaded_file, 'name', 'uploaded dataset')
    else:
        df = pd.read_csv(DATA_PATH, encoding='ISO-8859-1')
        source = str(DATA_PATH)

    text_col = infer_text_column(df)
    label_col = infer_label_column(df)

    if text_col is None or label_col is None:
        raise ValueError('Could not detect a valid text column or label column in the uploaded file.')

    df = df[[text_col, label_col]].rename(columns={text_col: 'message', label_col: 'label'})
    df = df.dropna(subset=['message']).reset_index(drop=True)
    df['message'] = df['message'].astype(str)
    df['label'] = normalize_label_values(df['label'])
    df = df[df['label'].isin(['ham', 'spam'])].reset_index(drop=True)

    return df, {
        'text_column': text_col,
        'label_column': label_col,
        'data_source': source
    }


def clean_text(message: str, remove_stopwords=True, apply_stemming=True):
    text = re.sub('[^a-zA-Z]', ' ', str(message)).lower().strip()
    tokens = [token for token in text.split() if token]
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    if apply_stemming:
        tokens = [STEMMER.stem(token) for token in tokens]
    return ' '.join(tokens)


@st.cache_data(show_spinner=False)
def prepare_features(corpus, max_features):
    vectorizer = CountVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X


def get_cleaned_corpus(df, remove_stopwords, apply_stemming):
    cleaned = df['message'].map(lambda t: clean_text(t, remove_stopwords, apply_stemming)).astype(str)
    valid_mask = cleaned.str.strip() != ''
    return cleaned, valid_mask


def get_sample_metrics(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion': confusion_matrix(y_test, y_pred)
    }


def render_header():
    st.set_page_config(page_title='Spam Detection Pipeline', page_icon='📊', layout='wide')
    st.markdown(PAGE_STYLE, unsafe_allow_html=True)
    st.markdown('<div class="bold-title">Interactive Spam Detection Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Explore the dataset, clean the text, choose features, train a model, and inspect performance - all in one dashboard.</p>', unsafe_allow_html=True)


def render_sidebar():
    st.sidebar.markdown('## 1. Data Source')
    uploaded_file = st.sidebar.file_uploader('Upload CSV', type=['csv'], key='uploaded_file', help='Upload your own spam dataset or use the default sample data.')
    if uploaded_file is not None:
        st.sidebar.success(f'Using uploaded file: {uploaded_file.name}')
    st.sidebar.markdown('---')
    st.sidebar.markdown('## 2. Model controls')
    selected_model = st.sidebar.selectbox('Choose model', list(MODEL_FACTORY.keys()))
    st.sidebar.markdown('### Cross-validation')
    use_cv = st.sidebar.checkbox('Enable 5-fold CV', value=True)
    st.sidebar.markdown('### Feature settings')
    features = st.sidebar.slider('Max vocabulary size', 1000, 6000, 4000, step=500)
    st.sidebar.markdown('---')
    st.sidebar.markdown('## 3. Training options')
    test_size = st.sidebar.slider('Test set size (%)', 10, 40, 20)
    return uploaded_file, selected_model, use_cv, features, test_size / 100.0


def render_data_eda(df, data_info):
    st.markdown('### Data & EDA')
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric('Total rows', len(df))
        st.metric('Spam rows', int((df['label'] == 'spam').sum()))
        st.metric('Ham rows', int((df['label'] == 'ham').sum()))
        st.markdown(f'**Text column:** {data_info.get("text_column", "unknown")}')
        st.markdown(f'**Label column:** {data_info.get("label_column", "unknown")}')
        st.markdown(f'**Data source:** {data_info.get("data_source", "default sample")}')
    with col2:
        st.subheader('Label distribution')
        dist = df['label'].value_counts().rename_axis('label').reset_index(name='count')
        st.bar_chart(data=dist, x='label', y='count')
    st.markdown('#### Dataset preview')
    st.dataframe(df[['label', 'message']].sample(min(5, len(df)), random_state=42).reset_index(drop=True), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_cleaning(df, remove_stopwords, apply_stemming):
    st.markdown('### Cleaning & Engineering')
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.write('Use the controls below to inspect how raw spam text is transformed before modeling.')
    st.write('- Remove punctuation and non-alphabetic characters')
    st.write('- Convert text to lowercase')
    st.write('- Remove stopwords and apply stemming')
    st.markdown('#### Raw vs cleaned sample')
    sample = df[['message']].sample(min(5, len(df)), random_state=11).reset_index(drop=True)
    sample['cleaned'] = sample['message'].map(lambda t: clean_text(t, remove_stopwords, apply_stemming))
    st.dataframe(sample, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_features(df, max_features, remove_stopwords, apply_stemming):
    st.markdown('### Feature Selection')
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.write('Select the vocabulary size used to convert messages into numeric features for the model.')
    cleaned, valid_mask = get_cleaned_corpus(df, remove_stopwords, apply_stemming)
    valid_count = int(valid_mask.sum())

    if valid_count == 0:
        st.error('No valid documents remain after cleaning. Try disabling stopword removal or applying less aggressive preprocessing.')
        st.markdown('</div>', unsafe_allow_html=True)
        return None

    corpus = cleaned[valid_mask].tolist()
    try:
        vectorizer, _ = prepare_features(corpus, max_features)
    except ValueError as exc:
        st.error('Feature extraction failed: ' + str(exc))
        st.markdown('</div>', unsafe_allow_html=True)
        return None

    top_features = sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1])[:20]
    st.markdown(f'**Top {min(20, len(top_features))} features:**')
    st.write([feature for feature, _ in top_features])

    if hasattr(vectorizer, 'get_feature_names_out'):
        features_list = vectorizer.get_feature_names_out()
    else:
        features_list = vectorizer.get_feature_names()
    st.markdown('#### Vocabulary sample')
    st.write(features_list[:20].tolist())
    st.markdown('</div>', unsafe_allow_html=True)
    return vectorizer


def render_training(df, selected_model_name, use_cv, max_features, test_size, remove_stopwords, apply_stemming):
    st.markdown('### Model Training')
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.write('Train the selected classifier on the cleaned spam dataset. Cross-validation helps estimate model stability.')
    st.write(f'**Selected model:** {selected_model_name}')
    st.write(f'**Max features:** {max_features}')
    st.write(f'**Test set size:** {int(test_size * 100)}%')
    st.write(f"**Cross-validation:** {'Enabled' if use_cv else 'Disabled'}")

    cleaned, valid_mask = get_cleaned_corpus(df, remove_stopwords, apply_stemming)
    valid_count = int(valid_mask.sum())
    if valid_count == 0:
        st.error('No valid training documents remain after cleaning. Adjust preprocessing or upload a different dataset.')
        st.markdown('</div>', unsafe_allow_html=True)
        return

    corpus = cleaned[valid_mask].tolist()
    try:
        vectorizer, X = prepare_features(corpus, max_features)
    except ValueError as exc:
        st.error('Feature extraction failed: ' + str(exc))
        st.markdown('</div>', unsafe_allow_html=True)
        return

    y = (df.loc[valid_mask, 'label'] == 'spam').astype(int).values
    if len(set(y)) < 2:
        st.error('The cleaned dataset contains only one class after filtering. Please upload a valid spam/ham dataset.')
        st.markdown('</div>', unsafe_allow_html=True)
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    if st.button('Start Training'):
        with st.spinner('Training the model...'):
            model = MODEL_FACTORY[selected_model_name]()
            if use_cv:
                scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=1)
            else:
                scores = None
            metrics = get_sample_metrics(model, X_train, X_test, y_train, y_test)
            st.session_state['training'] = {
                'model_name': selected_model_name,
                'vectorizer': vectorizer,
                'classifier': model,
                'metrics': metrics,
                'cv_scores': scores,
                'test_size': test_size,
                'remove_stopwords': remove_stopwords,
                'apply_stemming': apply_stemming
            }
            st.success('Model training complete!')

    if 'training' in st.session_state:
        training = st.session_state['training']
        st.markdown('#### Latest training run')
        st.write(f"Model: {training['model_name']}")
        if training['cv_scores'] is not None:
            st.write(f"CV accuracy: {training['cv_scores'].mean():.3f} ± {training['cv_scores'].std():.3f}")
        st.write('Click the Performance tab to review the latest evaluation metrics.')
    st.markdown('</div>', unsafe_allow_html=True)


def render_performance():
    st.markdown('### Performance')
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    if 'training' not in st.session_state:
        st.warning('Train a model first in the Model Training tab before viewing performance.')
        st.markdown('</div>', unsafe_allow_html=True)
        return

    metrics = st.session_state['training']['metrics']
    st.subheader('Evaluation metrics on holdout set')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Accuracy', f"{metrics['accuracy'] * 100:.1f}%")
    col2.metric('Precision', f"{metrics['precision'] * 100:.1f}%")
    col3.metric('Recall', f"{metrics['recall'] * 100:.1f}%")
    col4.metric('F1 Score', f"{metrics['f1'] * 100:.1f}%")

    st.markdown('#### Confusion matrix')
    cm = metrics['confusion']
    cm_df = pd.DataFrame(cm, index=['Actual Ham', 'Actual Spam'], columns=['Predicted Ham', 'Predicted Spam'])
    st.dataframe(cm_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    render_header()
    uploaded_file, selected_model, use_cv, max_features, test_size = render_sidebar()
    try:
        df, data_info = load_data(uploaded_file)
    except Exception as exc:
        st.error(str(exc))
        return

    if 'clean_settings' not in st.session_state:
        st.session_state['clean_settings'] = {'remove_stopwords': True, 'apply_stemming': True}

    tabs = st.tabs(['Data & EDA', 'Cleaning & Engineering', 'Feature Selection', 'Model Training', 'Performance'])

    with tabs[0]:
        render_data_eda(df, data_info)

    with tabs[1]:
        remove_stopwords = st.checkbox('Remove stopwords', value=st.session_state['clean_settings']['remove_stopwords'])
        apply_stemming = st.checkbox('Apply stemming', value=st.session_state['clean_settings']['apply_stemming'])
        st.session_state['clean_settings']['remove_stopwords'] = remove_stopwords
        st.session_state['clean_settings']['apply_stemming'] = apply_stemming
        render_cleaning(df, remove_stopwords, apply_stemming)

    with tabs[2]:
        render_features(df, max_features, st.session_state['clean_settings']['remove_stopwords'], st.session_state['clean_settings']['apply_stemming'])

    with tabs[3]:
        render_training(df, selected_model, use_cv, max_features, test_size, st.session_state['clean_settings']['remove_stopwords'], st.session_state['clean_settings']['apply_stemming'])

    with tabs[4]:
        render_performance()


if __name__ == '__main__':
    main()
