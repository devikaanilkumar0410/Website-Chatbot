import requests
from bs4 import BeautifulSoup

urls = [
    "https://luminartechnolab.com/",
    "https://luminartechnolab.com/course",
    "https://luminartechnolab.com/about"
    # "https://www.luminartechnolab.com/course-detail/machine-learning-training-kochi",
    # "https://www.luminartechnolab.com/course-detail/python-data-analytics-with-power-bI",
    # "https://www.luminartechnolab.com/course-detail/python-training-kochi",
    # ""

]

all_text = ""

for url in urls:
    print(f"Scraping {url}")

    try:
        response = requests.get(url)

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        text = soup.get_text(
            separator="\n",
            strip=True
        )

        all_text += text + "\n"

    except Exception as e:
        print(e)

with open(
    "data/luminar.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(all_text)

print("Data saved")