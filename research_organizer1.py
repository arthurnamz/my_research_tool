import streamlit as st
import PyPDF2
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from collections import Counter
from st_link_analysis import st_link_analysis, NodeStyle, EdgeStyle

# Extract text and keywords from a PDF
def extract_text_and_keywords(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), top_n=10)
    return text, [kw[0] for kw in keywords]

# Compute similarity matrix
def compute_similarity(texts):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(texts)
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix, vectors

# Perform clustering
def cluster_papers(vectors, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(vectors)
    return labels

# Infer the main domain for each cluster
def infer_cluster_domains(paper_keywords, labels):
    cluster_domains = {}
    for cluster in set(labels):
        # Collect all keywords from papers in the same cluster
        cluster_keywords = [
            keyword
            for paper_idx, cluster_label in enumerate(labels)
            if cluster_label == cluster
            for keyword in paper_keywords[paper_idx]
        ]
        # Determine the most common keyword as the domain
        if cluster_keywords:
            main_domain = Counter(cluster_keywords).most_common(1)[0][0]
        else:
            main_domain = f"Cluster {cluster}"
        cluster_domains[cluster] = main_domain
    return cluster_domains

# Create JSON structured data for network visualization
def create_network_json(titles, keywords, similarity_matrix, labels, cluster_domains, threshold=0.5):
    elements = {"nodes": [], "edges": []}
    node_id_map = {}

    # Create cluster nodes
    for cluster_id, domain in cluster_domains.items():
        elements["nodes"].append({
            "data": {
                "id": f"cluster-{cluster_id}",
                "label": "CLUSTER",
                "name": domain,
                "keywords": domain,
                "cluster": cluster_id,
            }
        })

    # Create paper nodes and connect to clusters
    for i, (title, keyword_list) in enumerate(zip(titles, keywords)):
        elements["nodes"].append({
            "data": {
                "id": int(i + 1),
                "label": "PAPER",
                "name": title,
                "keywords": ", ".join(keyword_list),
                "cluster": labels[i],
            }
        })
        node_id_map[title] = int(i + 1)

        # Add an edge to the main cluster
        elements["edges"].append({
            "data": {
                "id": f"edge-cluster-{i + 1}",
                "label": "BELONGS_TO",
                "source": int(i + 1),
                "target": f"cluster-{labels[i]}",
            }
        })

        # Add edges to other clusters if the paper belongs to multiple domains
        for other_cluster_id, domain in cluster_domains.items():
            if other_cluster_id != labels[i] and any(kw in domain for kw in keyword_list):
                elements["edges"].append({
                    "data": {
                        "id": f"edge-cross-cluster-{i + 1}-{other_cluster_id}",
                        "label": "CROSS_DOMAIN",
                        "source": int(i + 1),
                        "target": f"cluster-{other_cluster_id}",
                    }
                })

    # Create edges between similar papers based on similarity matrix
    edge_id = 1
    for i in range(len(similarity_matrix)):
        for j in range(i + 1, len(similarity_matrix)):
            if similarity_matrix[i][j] > threshold:
                elements["edges"].append({
                    "data": {
                        "id": f"edge-similarity-{edge_id}",
                        "label": "SIMILARITY",
                        "source": node_id_map[titles[i]],
                        "target": node_id_map[titles[j]],
                        "weight": float(similarity_matrix[i][j]),
                    }
                })
                edge_id += 1

    return elements

# Streamlit App
def main():
    st.title("Dynamic Research Paper Network with Domains")
    st.write("Upload research papers, analyze their relationships, and visualize the connections dynamically.")

    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])

    if uploaded_files:
        paper_texts = []
        paper_titles = []
        paper_keywords = []

        for file in uploaded_files:
            text, keywords = extract_text_and_keywords(file)
            title = file.name.split(".")[0]
            paper_texts.append(text)
            paper_titles.append(title)
            paper_keywords.append(keywords)

        similarity_matrix, vectors = compute_similarity(paper_texts)
        n_clusters = st.slider("Number of Clusters", 2, 10, 3)
        labels = cluster_papers(vectors, n_clusters)
        threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.5)

        # Infer main domains for clusters
        cluster_domains = infer_cluster_domains(paper_keywords, labels)

        # Create JSON structure for visualization
        elements = create_network_json(paper_titles, paper_keywords, similarity_matrix, labels, cluster_domains, threshold)

        # Define node and edge styles
        node_styles = [
            NodeStyle("CLUSTER", "#309A60", "name", "cloud"),
            NodeStyle("PAPER", "#2A629B", "name", "person"),
        ]
        edge_styles = [
            EdgeStyle("BELONGS_TO", labeled=False, directed=True),
            EdgeStyle("CROSS_DOMAIN", labeled=True, directed=True),
            EdgeStyle("SIMILARITY", labeled=True, directed=False),
        ]

        # Define layout
        layout = {"name": "cose", "animate": "end", "nodeDimensionsIncludeLabels": False}

        # Visualize the network using st_link_analysis
        st.write("Network Visualization:")
        st_link_analysis(elements, node_styles=node_styles, edge_styles=edge_styles, layout=layout, key="network")

        st.write("Link Analysis Complete.")

if __name__ == "__main__":
    main()
