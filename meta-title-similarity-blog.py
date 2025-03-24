import pandas as pd
import requests
import gdown
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



# Initialize the Sentence-BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to fetch the content from a URL and extract text
def fetch_article_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Extract the <h1> and <p> tags with the specified class
        h1_text = soup.find('h1').get_text(strip=True) if soup.find('h1') else ''
        article_text = " ".join([p.get_text(strip=True) for p in soup.find_all('p', class_='pb-14 relative article-content')])
        print(article_text)
        # Combine the header and article content
        full_text = h1_text + " " + article_text
        return full_text.strip()
    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        return None

# Read the Excel file containing URLs and Titles

url = "https://docs.google.com/spreadsheets/d/1j9fsTrFv3rgHj9aJgUe1LCZn-lRAQdQKsQE28gYJIqw/edit?usp=sharing"
output = "urls_metaTitle.xlsx"
gdown.download(url, output, quiet=False)
df = pd.read_excel(output, engine="openpyxl")

# Ensure the columns are correctly named in the Excel file
if 'Address' not in df.columns or 'Title' not in df.columns:
    raise ValueError("Excel file must contain columns named 'Address' and 'Title'.")

# List to store embeddings for all URLs and Titles
url_embeddings = []
title_embeddings = []

# Iterate over all rows, fetch the content, and embed both URL content and Title
for idx, row in df.iterrows():
    url = row['Address']  # Using 'Address' as per the requirement
    title = row['Title']
    print(f"Processing URL: {url}")

    # Fetch content and embed it
    content = fetch_article_content(url)
    if content:
        # Embed the full article content
        content_embedding = model.encode([content])  # Embed the full content
        url_embeddings.append(content_embedding[0])  # Append the first (and only) embedding
    else:
        url_embeddings.append(None)
    
    # Embed the title
    if title:
        title_embedding = model.encode([title])  # Embed the title text
        title_embeddings.append(title_embedding[0])  # Append the first (and only) embedding
    else:
        title_embeddings.append(None)

# Add the embeddings as new columns to the DataFrame
df['Embeddings'] = url_embeddings
df['Title Embedding'] = title_embeddings

# Function to calculate similarity for a new title
def calculate_similarity(new_title, content_embedding):
    title_embedding = model.encode([new_title])[0]  # Embed the new title
    similarity = cosine_similarity([content_embedding], [title_embedding])[0][0]
    print(f"Similarity score between the new title and the content: {similarity:.4f}")
    
    # Display if it's "work" or "good" or 'very good'
    if similarity < 0.8:
        print("This similarity is categorized as: more work")
    elif  similarity < 0.9:
        print("This similarity is categorized as: good")
    else:
        print("This similarity is categorized as: very good")

# Main function to interact with the user
def interact_with_content():
    while True:
        print("\nChoose a title from the following list:")
        for idx, title in enumerate(df['Title']):
            print(f"{idx + 1}. {title}")
        
        # Let the user select a title
        selected_index = int(input("Enter the number of the title you want to work with (or 0 to exit): ")) - 1
        
        if selected_index == -1:
            print("Exiting the program.")
            break
        
        if selected_index < 0 or selected_index >= len(df):
            print("Invalid selection. Please choose a valid title number.")
            continue
        
        # Get the chosen row
        selected_row = df.iloc[selected_index]
        url = selected_row['Address']
        title = selected_row['Title']
        content_embedding = selected_row['Embeddings']
        
        print(f"\nYou selected the title: '{title}'")
        print(f"Fetching content from URL: {url}")
        
        # Calculate the similarity for the initial title
        print("\nCalculating similarity for the selected title and its content...")
        calculate_similarity(title, content_embedding)
        
        # Allow the user to rewrite the title and get the similarity score
        while True:
            new_title = input("\nEnter a new title to compare its similarity with the content (or type 'exit' to quit): ")
            if len(new_title) > 60:
                print("Your Meta title is too long ! take off ", len(new_title)-60, "characters from it !")

            if new_title.lower() == 'exit':
                print("Exiting the title rewrite section.")
                break
            
            # Calculate similarity for the new title
            calculate_similarity(new_title, content_embedding)

# Start the interaction
interact_with_content()
