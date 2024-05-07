import requests
import os

def download_file(url, filename):
    response = requests.get(url)
    response.raise_for_status()  # Check if the download was successful
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded {filename}")

# URL for Greek to English Tatoeba dataset
url = "https://object.pouta.csc.fi/OPUS-Tatoeba/v2021-12-21/moses/el-en.txt.zip"
filename = "el-en.txt.zip"

# Download the file
download_file(url, filename)
