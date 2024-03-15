from Logic.core.crawler import IMDbCrawler
from bs4 import BeautifulSoup


a = IMDbCrawler()
URL = 'https://www.imdb.com/title/tt0060196/'
response = a.crawl(URL)
parsed = BeautifulSoup(response.text, 'html.parser')
movie = a.get_imdb_instance()
a.extract_movie_info(parsed ,movie, URL)
print(movie)