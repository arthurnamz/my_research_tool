import os
import fitz  # PyMuPDF is used for working with PDFs
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Define a function to extract text from a PDF

def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    extracted_text = ""

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        text = page.get_text()
        extracted_text += text

    pdf_document.close()
    return extracted_text

# Preprocess text by removing stopwords and punctuation
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [
        word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words
    ]
    return filtered_tokens

# Extract keywords from the text
def extract_keywords(text):
    # Assume keywords are under a section labeled 'Keywords' or similar
    keywords_section = "".join(
        line for line in text.splitlines() if 'keywords' in line.lower()
    )
    keywords = word_tokenize(keywords_section)
    return [word.lower() for word in keywords if word.isalnum()]

# Compare keywords and most common words in two documents
def compare_documents(text1, text2):
    # Extract keywords
    keywords1 = set(extract_keywords(text1))
    keywords2 = set(extract_keywords(text2))

    # Find common keywords
    common_keywords = keywords1 & keywords2

    # Preprocess the text to find common words
    words1 = preprocess_text(text1)
    words2 = preprocess_text(text2)

    # Find the most common words in both documents
    common_words = set(Counter(words1).keys()) & set(Counter(words2).keys())

    return common_keywords, common_words

# Visualize relationships using a graph
def visualize_relationships(common_keywords, common_words):
    G = nx.Graph()

    # Add nodes for keywords and words
    for keyword in common_keywords:
        G.add_node(keyword, color='red')
    for word in common_words:
        G.add_node(word, color='blue')

    # Add edges to connect common keywords and words (can be customized)
    for keyword in common_keywords:
        for word in common_words:
            G.add_edge(keyword, word)

    # Draw the graph
    colors = [G.nodes[node]['color'] for node in G.nodes()]
    nx.draw(G, with_labels=True, node_color=colors)
    plt.show()

# Main function to process two PDFs
def process_and_compare(pdf1_path, pdf2_path):
    # Extract text from both PDFs
    text1 = extract_text_from_pdf(pdf1_path)
    text2 = extract_text_from_pdf(pdf2_path)

    # Compare the documents
    common_keywords, common_words = compare_documents(text1, text2)

    # Display the results
    print("Common Keywords:", common_keywords)
    print("Common Words:", common_words)

    # Visualize the relationships
    visualize_relationships(common_keywords, common_words)

# Example usage
pdf1_path = "../files/pdfs/a.pdf"  # Replace with your PDF file path
pdf2_path = "../files/pdfs/b.pdf"  # Replace with your PDF file path

process_and_compare(pdf1_path, pdf2_path)
