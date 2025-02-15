import os
import fitz  # PyMuPDF is used for working with PDFs

# Define a function to extract text from a PDF and save it to a .txt file
def extract_text_from_pdf(pdf_path, txt_path):
    """
    Extracts text from a PDF file and saves it into a .txt file.
    Parameters:
        pdf_path (str): Path to the input PDF file.
        txt_path (str): Path to save the output text file.
    """
    # Open the PDF file using fitz
    pdf_document = fitz.open(pdf_path)  # Load the PDF into memory
    extracted_text = ""  # Initialize a variable to hold all the text from the PDF

    # Loop through all the pages in the PDF
    for page_num in range(len(pdf_document)):  # len(pdf_document) gives the number of pages
        page = pdf_document[page_num]  # Access a specific page by its index
        text = page.get_text()  # Extract the text content of the page
        extracted_text += text  # Append the text from the current page to the overall text

    # Close the PDF file to free up resources
    pdf_document.close()

    # Save the extracted text into a .txt file
    with open(txt_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(extracted_text)  # Write the text into the file

    # Print a message indicating that the process is complete
    print(f"Text extracted from {pdf_path} and saved to {txt_path}")

# Define a function to handle multiple PDF files
def process_multiple_pdfs(pdf_directory, output_directory):
    """
    Processes multiple PDF files in a directory and extracts text from each.
    Parameters:
        pdf_directory (str): Directory containing the input PDF files.
        output_directory (str): Directory to save the output .txt files.
    """
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Iterate over all files in the PDF directory
    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):  # Process only PDF files
            pdf_path = os.path.join(pdf_directory, filename)  # Full path to the PDF file
            txt_filename = f"{os.path.splitext(filename)[0]}.txt"  # Replace .pdf with .txt
            txt_path = os.path.join(output_directory, txt_filename)  # Full path to the output .txt file

            # Extract text from the current PDF file and save it
            extract_text_from_pdf(pdf_path, txt_path)

# Specify the input and output directories
pdf_directory = "../files/pdfs"  # Replace with the directory containing your PDF files
output_directory = "../files/txt_files"  # Replace with the directory to save output .txt files

# Call the function to process multiple PDFs
process_multiple_pdfs(pdf_directory, output_directory)





