import requests
from bs4 import BeautifulSoup

# Hedef URL
url = ""

# Sayfayı çekme
response = requests.get(url)
response.encoding = response.apparent_encoding  # Sayfanın gerçek encoding'ini algıla

# HTML içeriğini parse etme
soup = BeautifulSoup(response.text, "html.parser")

# Fıkraları içeren <p> etiketlerini bulma
jokes = []
for p in soup.find_all("p"):
    joke_text = p.get_text(strip=True)
    if joke_text:  # Boş olmayanları al
        jokes.append(joke_text)

# Sonuçları ekrana yazdırma
for idx, joke in enumerate(jokes, 1):
    print(f"Fıkra {idx}:\n{joke}\n")

with open("fikra.txt", "w", encoding="utf-8") as f:
    for joke in jokes:
        f.write(joke + "\n\n")