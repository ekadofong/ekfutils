import requests
from bs4 import BeautifulSoup

def fetch(url, session=None):
    if session is None:
        session = requests    
    response = session.get(url)
    html_content = response.text
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')    
    return soup