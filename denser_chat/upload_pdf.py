import os

def ensure_upload_folder(folder="uploadedPDF"):
    """
    Checks if the folder exists, and if not, creates it.
    Parameters:
      - folder (str): The folder name/path to check/create. Defaults to 'uploadedPDF'.
    Returns:
      - str: The path to the folder.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")
    else:
        print(f"Folder already exists: {folder}")
    return folder

def append_file_to_sources(file_path, sources_file="sources.txt"):
    """
    Appends the given file path to the sources file on a new line, ensuring forward slashes.
    Parameters:
      - file_path (str): The relative path to the file to be added.
      - sources_file (str): The file where the path should be appended. Defaults to 'sources.txt'.
    """
    file_path = file_path.replace("\\", "/")
    with open(sources_file, "a", encoding="utf-8") as f:
        f.write(file_path + "\n")
    print(f"Appended file path to {sources_file}: {file_path}")

if __name__ == "__main__":
    folder = ensure_upload_folder()
    sample_path = os.path.join(".", folder, "example.pdf")
    append_file_to_sources(sample_path)
