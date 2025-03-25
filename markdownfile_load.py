import os

def load_markdown_files(directory):
    documents = []
    for file in os.listdir(directory):
        if file.endswith(".md"):  # Check for markdown files
            with open(os.path.join(directory, file), "r", encoding="utf-8") as f:
                documents.append(f.read())
    return documents

# # Load all markdown files from a folder
# docs = load_markdown_files("data")
# print(docs)
