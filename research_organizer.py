import streamlit as st
import os
import PyPDF2
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from pyvis.network import Network 
import matplotlib.pyplot as plt
from st_link_analysis import st_link_analysis, NodeStyle, EdgeStyle

# Extract text and keywords from a PDF
def extract_text_and_keywords(pdf_file):
    # Reading the PDF
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    # Extract keywords using KeyBERT
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), top_n=10)
    return text, [kw[0] for kw in keywords]

# Compute similarity matrix
def compute_similarity(texts):
    # Create a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(texts)
    
    # Compute cosine similarity between texts
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix, vectors

# Perform clustering
def cluster_papers(vectors, n_clusters=3):
    # Clustering with KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(vectors)
    return labels

# Create interactive canvas for network visualization
def create_interactive_canvas(titles, similarity_matrix, labels, threshold=0.5):
    # Initialize the PyVis Network
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black", notebook=True)
    
    # Generate cluster colors (cast label to int before using for color assignment)
    cluster_colors = {label: f"#{''.join([hex(int(c))[2:].zfill(2) for c in plt.cm.tab10(int(label) % 10)[:3]])}" for label in set(labels)}
    
    # Add nodes with cluster colors and size based on degree
    for i, title in enumerate(titles):
        net.add_node(i, label=title, color=cluster_colors[int(labels[i])], title=title, size=len([j for j in range(len(similarity_matrix)) if similarity_matrix[i][j] > threshold]) * 10)
    
    # Add edges for similar papers based on the similarity matrix
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i][j] > threshold:
                net.add_edge(i, j, value=similarity_matrix[i][j])
    
    # Set physics for a more dynamic layout
    net.force_atlas_2based()
    
    # Generate and save the interactive network visualization
    network_html = "network.html"
    net.show(network_html)
    return network_html

# Streamlit App
def main():
    st.title("Enhanced Research Paper Organizer")
    st.write("Upload research papers, organize them into clusters, and visualize their relationships.")
    
    # File uploader
    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
    
    if uploaded_files:
        paper_texts = []
        paper_titles = []
        
        # Process each uploaded file
        for file in uploaded_files:
            text, keywords = extract_text_and_keywords(file)
            title = file.name.split(".")[0]  # Extract title from filename
            paper_texts.append(text)
            paper_titles.append(title)
        
        # Compute the similarity matrix and vector representation
        similarity_matrix, vectors = compute_similarity(paper_texts)
        
        # Allow the user to select the number of clusters
        n_clusters = st.slider("Number of Clusters", 2, 10, 3)
        labels = cluster_papers(vectors, n_clusters)
        
        # Allow the user to set a similarity threshold
        threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.5)
        
        # Create the interactive graph visualization
        network_html = create_interactive_canvas(paper_titles, similarity_matrix, labels, threshold)
        
        # Display network in Streamlit using an iframe
        st.markdown(f'<iframe src="{network_html}" width="100%" height="800"></iframe>', unsafe_allow_html=True)
        
        # Allow download of the network visualization as an HTML file
        with open(network_html, "rb") as file:
            st.download_button(
                label="Download Network Graph",
                data=file,
                file_name="network.html",
                mime="text/html"
            )
        
        # Link Analysis with st-link-analysis
        st.write("Perform Link Analysis:")
        st_link_analysis(paper_titles, similarity_matrix, labels, threshold)  # Updated to use st_link_analysis

if __name__ == "__main__":
    main()
