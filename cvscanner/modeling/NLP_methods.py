import pandas as pd
import json , re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.offline as pyo
import plotly.graph_objs as go
pyo.init_notebook_mode(connected=True)
    

def load_category_data(json_file_path):
    """Load the categorized resumes from JSON file"""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def prepare_training_data(category_data):
    """
    Prepare training data for TF-IDF vectorizer
    Returns: documents, category labels, and category to document mapping
    """
    documents = []
    categories = []
    category_docs = defaultdict(list)
    
    for category, resumes in category_data.items():
        for resume in resumes:
            documents.append(resume)
            categories.append(category)
            category_docs[category].append(resume)
    
    return documents, categories, category_docs

def find_top_categories(text_cv, json_file_path, top_n=5):
    """
    Find top N most similar categories to the given CV text
    
    Args:
        text_cv (str): The CV text to compare
        json_file_path (str): Path to the JSON file with categorized resumes
        top_n (int): Number of top categories to return
    
    Returns:
        list: Top N categories with their similarity scores
    """
    
    # Load category data
    category_data = load_category_data(json_file_path)
    
    # Prepare training data
    documents, categories, category_docs = prepare_training_data(category_data)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    
    # Fit and transform the training documents
    X = vectorizer.fit_transform(documents)
    
    # Transform the input CV text
    cv_vector = vectorizer.transform([text_cv])
    
    # Calculate cosine similarity between CV and all documents
    similarities = cosine_similarity(cv_vector, X).flatten()
    
    # Calculate average similarity per category
    category_scores = defaultdict(list)
    for i, similarity in enumerate(similarities):
        category_scores[categories[i]].append(similarity)
    
    # Get average similarity for each category
    category_avg_scores = {}
    for category, scores in category_scores.items():
        category_avg_scores[category] = np.mean(scores)
    
    # Sort categories by average similarity (descending)
    sorted_categories = sorted(
        category_avg_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:top_n]
    
    # Return top N categories
    result = []
    for category, score in sorted_categories:
        percentage = round(score * 100, 2)  # Convert to percentage and round to 2 decimals
        result.append((category, percentage))
    
    return result,similarities,vectorizer

def find_top_categories_alternative(text_cv, json_file_path, top_n=5):
    """
    Alternative approach: Create representative text for each category
    and compare with input CV
    """
    
    category_data = load_category_data(json_file_path)
    
    # Create representative text for each category (concatenate all resumes)
    category_representatives = {}
    for category, resumes in category_data.items():
        # Combine all resumes in the category with some separation
        representative_text = " ".join(resumes)
        category_representatives[category] = representative_text
    
    # Prepare documents for vectorization
    documents = list(category_representatives.values())
    category_names = list(category_representatives.keys())
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    # Fit and transform category representative texts
    X = vectorizer.fit_transform(documents)
    
    # Transform the input CV text
    cv_vector = vectorizer.transform([text_cv])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(cv_vector, X).flatten()
    
    # Pair categories with their similarity scores
    category_scores = list(zip(category_names, similarities))
    
    # Sort by similarity (descending)
    category_scores.sort(key=lambda x: x[1], reverse=True)
    
    return category_scores[:top_n]

def preprocess_text(text):
    """Enhanced text preprocessing"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def enhanced_find_top_categories(text_cv, json_file_path, top_n=5):
    """Enhanced version with better text preprocessing"""
    
    category_data = load_category_data(json_file_path)
    
    # Preprocess all texts
    preprocessed_cv = preprocess_text(text_cv)
    
    category_representatives = {}
    for category, resumes in category_data.items():
        # Preprocess and combine resumes
        preprocessed_resumes = [preprocess_text(resume) for resume in resumes]
        representative_text = " ".join(preprocessed_resumes)
        category_representatives[category] = representative_text
    
    # Vectorize
    vectorizer = TfidfVectorizer(
        max_features=6000,
        stop_words='english',
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.85
    )
    
    documents = list(category_representatives.values())
    category_names = list(category_representatives.keys())
    
    X = vectorizer.fit_transform(documents)
    cv_vector = vectorizer.transform([preprocessed_cv])
    
    similarities = cosine_similarity(cv_vector, X).flatten()
    
    category_scores = list(zip(category_names, similarities))
    category_scores.sort(key=lambda x: x[1], reverse=True)
    
    return category_scores[:top_n]

def print_formatted_results(results):
    """Print results in a nice formatted way"""
    percentage_results = []
    for i, (category, score) in enumerate(results, 1):
        percentage = round(float(score) * 100, 2)
        percentage_results.append((category, percentage))
        print(f"{i}. {category}: {percentage}%")
    
    return percentage_results



def visualize_cv_in_trained_space(similarities, documents, categories, input_cv_text, vectorizer=None):
    """
    Visualize input CV position within trained CV space using dimensionality reduction
    
    Args:
        similarities: Cosine similarity scores between input CV and all training documents
        documents: List of all training CV texts
        categories: List of categories for each training document
        input_cv_text: The input CV text to visualize
        vectorizer: Optional pre-fitted vectorizer (if None, creates new one)
    """
    
    # Import plotly offline for Jupyter notebook display

    # Create or use existing vectorizer
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        training_vectors = vectorizer.fit_transform(documents)
    else:
        training_vectors = vectorizer.transform(documents)
    
    # Transform input CV using the same vectorizer
    input_cv_vector = vectorizer.transform([input_cv_text])
    
    # Combine all vectors for consistent dimensionality reduction
    all_vectors = np.vstack([training_vectors.toarray(), input_cv_vector.toarray()])
    
    # Apply dimensionality reduction
    print("Reducing dimensions with t-SNE...")
    
    # First reduce with PCA for better t-SNE performance
    pca = PCA(n_components=min(50, all_vectors.shape[1]))
    reduced_pca = pca.fit_transform(all_vectors)
    
    # Use t-SNE for final 2D visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(documents)-1))
    reduced_2d = tsne.fit_transform(reduced_pca)
    
    # Separate training points and input CV point
    training_points = reduced_2d[:-1]  # All except last point
    input_cv_point = reduced_2d[-1]    # Last point is input CV
    
    # Create DataFrame for training data
    plot_data = pd.DataFrame({
        'x': training_points[:, 0],
        'y': training_points[:, 1],
        'category': categories,
        'similarity': similarities,
        'hover_text': [f"Category: {cat}<br>Similarity: {sim:.3f}" 
                      for cat, sim in zip(categories, similarities)],
        'size': [sim * 20 + 5 for sim in similarities]  # Size based on similarity
    })
    
    # Create the main scatter plot for training data
    fig = px.scatter(
        plot_data, 
        x='x', 
        y='y',
        color='category',
        hover_data={'hover_text': True, 'x': False, 'y': False, 'category': False},
        size='size',
        title='Input CV Position in Trained CV Space',
        labels={'x': 't-SNE Dimension 1', 'y': 't-SNE Dimension 2'},
        opacity=0.7
    )
    
    # Add the input CV as a special point
    fig.add_trace(go.Scatter(
        x=[input_cv_point[0]],
        y=[input_cv_point[1]],
        mode='markers+text',
        marker=dict(
            size=20,
            color='black',
            symbol='star',
            line=dict(width=2, color='white')
        ),
        name='Input CV',
        hoverinfo='text',
        hovertext='Your Input CV',
        text='Your CV',
        textposition='middle center'
    ))
    
    # Update layout
    fig.update_layout(
        hovermode='closest',
        showlegend=True,
        width=900,
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    print("Visualization ready! Showing plot...")
    
    # Use offline iplot for Jupyter notebook display
    pyo.iplot(fig)
    
    return fig, reduced_2d

