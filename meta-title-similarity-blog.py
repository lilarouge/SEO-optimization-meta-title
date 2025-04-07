import pandas as pd
import requests
import gdown
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



# Initialize the Sentence-BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Read the Excel file containing URLs and Titles

url = "https://docs.google.com/spreadsheets/d/1j9fsTrFv3rgHj9aJgUe1LCZn-lRAQdQKsQE28gYJIqw/edit?usp=sharing"
output = "urls_metaTitle.xlsx"
gdown.download(url, output, quiet=False)
df = pd.read_excel(output, engine="openpyxl")

# Ensure the columns are correctly named in the Excel file
if 'Address' not in df.columns or 'Title' not in df.columns:
    raise ValueError("Excel file must contain columns named 'Address' and 'Title'.")


# Function to fetch the content from a URL and extract text
def fetch_article_content(url):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error {response.status_code} for URL: {url}")
            return "Error fetching content"

        title_content_raw = response.text

        title_match = re.search(r'<h1 class="mb-4 mt-0 lg:pr-4 ">(.*?)</h1>', title_content_raw)
        content_match = re.search(r'<div class="pb-14 relative article-content">(.*?)</div>', title_content_raw, re.DOTALL)

        if title_match and content_match:
            title = title_match.group(1).strip().lower()
            article_html = content_match.group(1)

            soup = BeautifulSoup(article_html, "html.parser")
            raw_text = soup.get_text(separator="\n")
            clean_text = re.sub(r"\s+", " ", raw_text).strip().lower()

            return title + "\n" + clean_text
        else:
            return "Missing title or content"
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return "Error fetching content"

# Function to fetch the top search query for a URL
def query_top_query(client: Resource, url: str) -> str:
    payload = {
        "startDate": "2024-01-31",
        "endDate": datetime.now().strftime("%Y-%m-%d"),
        "dimensions": ["query"],
        "dimensionFilterGroups": [
            {
                "filters": [
                    {
                        "dimension": "page",
                        "operator": "equals",
                        "expression": url
                    }
                ]
            }
        ],
        "rowLimit": 200
    }

    response = client.searchanalytics().query(siteUrl=DOMAIN, body=payload).execute()

    if "rows" in response and response["rows"]:
        rows = response["rows"]
        rows_sorted_by_clicks = sorted(rows, key=lambda x: x.get("clicks", 0), reverse=True)

        if rows_sorted_by_clicks[0].get("clicks", 0) < 20:
            rows_sorted_by_clicks = sorted(rows, key=lambda x: x.get("impressions", 0), reverse=True)

        for row in rows_sorted_by_clicks:
            top_query = row["keys"][0].lower()
            if "atera" not in top_query:
                cleaned_query = re.sub(r'[^a-zA-Z0-9\s]', '', top_query).strip()
                return cleaned_query

    return "No valid query found"

# Ensure the columns are correctly named
if 'Address' not in df.columns or 'Title' not in df.columns:
    raise ValueError("Excel file must contain columns named 'Address' and 'Title'.")

# Pre-embed only titles
df['Title Embedding'] = df['Title'].apply(lambda t: model.encode([t])[0] if pd.notnull(t) else None)

# Function to calculate similarity
def calculate_similarity(new_title, content_embedding):
    title_embedding = model.encode([new_title])[0]
    similarity = cosine_similarity([content_embedding], [title_embedding])[0][0]
    print(f"Similarity score between the new title and the content from 0 to 1: {similarity:.4f}")

    if similarity < 0.8:
        print("This similarity is categorized as: more work")
    elif similarity < 0.9:
        print("This similarity is categorized as: good")
    else:
        print("This similarity is categorized as: very good")

# Interactive function
def interact_with_content():
    while True:
        print("\nChoose a title from the following list:")
        for idx, title in enumerate(df['Title']):
            print(f"{idx + 1}. {title}")

        try:
            selected_index = int(input("Enter the number of the title you want to work with (or 0 to exit): ")) - 1
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue

        if selected_index == -1:
            print("Exiting the program.")
            break

        if selected_index < 0 or selected_index >= len(df):
            print("Invalid selection. Please choose a valid title number.")
            continue

        selected_row = df.iloc[selected_index]
        url = selected_row['Address']
        title = selected_row['Title']

        print(f"\nYou selected the title: '{title}'")
        print(f"Fetching content from URL: {url}...")

        content = fetch_article_content(url)
        #print(content)
        if content == "Error fetching content" or content == "Missing title or content":
            print(" Could not fetch content for this URL.")
            continue

        content_embedding = model.encode([content])[0]

        top_query = query_top_query(service, url)

        print("\nCalculating similarity for the selected title and its content...")
        calculate_similarity(title, content_embedding)

        while True:
            print(f"The top query for this page is: {top_query}. Use it in your new meta title!")
            new_title = input("\nEnter a new title (or type 'exit' to quit): ")
            if new_title.lower() == 'exit':
                print("Exiting this title.")
                break
            if len(new_title) > 60:
                print(f" Your Meta title is too long! Remove {len(new_title)-60} characters.")
            calculate_similarity(new_title, content_embedding)

# Start the interaction
interact_with_content()
