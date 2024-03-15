from requests import get
from bs4 import BeautifulSoup
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
import json
import re


class IMDbCrawler:
    """
    put your own user agent in the headers
    """
    headers = {
        # "Host": "www.imdb.com",
"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
'Accept-Language': 'en-US,en;q=0.9',
# "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
# "Accept-Language": "en-US,en;q=0.5",
# "Accept-Encoding": "gzip, deflate, br",
# Alt-Used: www.imdb.com
# Connection: keep-alive
# "Cookie": "session-id=140-3113877-4015914; session-id-time=2082787201l; csm-hit=tb:s-EZ0Y5W37E0GXAXFJ868H|1710250879864&t:1710250881412&adb:adblk_no; ad-oo=0; ci=e30",
# Upgrade-Insecure-Requests: 1
# Sec-Fetch-Dest: document
# Sec-Fetch-Mode: navigate
# Sec-Fetch-Site: none
# Sec-Fetch-User: ?1
    }
    top_250_URL = 'https://www.imdb.com/chart/top/'

    def __init__(self, crawling_threshold=1000):
        """
        Initialize the crawler

        Parameters
        ----------
        crawling_threshold: int
            The number of pages to crawl
        """
        # TODO
        self.crawling_threshold = crawling_threshold
        self.not_crawled = deque()
        self.crawled = []
        self.added_ids = set()
        self.add_list_lock = Lock()
        self.add_queue_lock = Lock()

    def get_id_from_URL(self, URL):
        """
        Get the id from the URL of the site. The id is what comes exactly after title.
        for example the id for the movie https://www.imdb.com/title/tt0111161/?ref_=chttp_t_1 is tt0111161.

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        str
            The id of the site
        """
        # print(URL)
        # print('dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd')
        if URL.startswith('http'):
            return URL.split('/')[4]
        else:
            return URL.split('/')[2]

    def write_to_file_as_json(self):
        """
        Save the crawled files into json
        """
        print(len(self.crawled))
        
        with open('IMDB_crawled.json', 'w') as f:
            f.write(json.dumps(list(self.crawled)))
            

        with open('IMDB_not_crawled.json', 'w') as f:
            f.write(json.dumps(list(self.not_crawled)))


    def read_from_file_as_json(self):
        """
        Read the crawled files from json
        """
        # TODO
        with open('IMDB_crawled.json', 'r') as f:
            self.crawled = None

        with open('IMDB_not_crawled.json', 'r') as f:
            self.not_crawled = None

        self.added_ids = None

    def crawl(self, URL):
        """
        Make a get request to the URL and return the response

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        requests.models.Response
            The response of the get request
        """
        resp = get(url=URL, headers=self.headers)
        while(resp.status_code == 504):
            resp = get(url=URL, headers=self.headers)

        if resp.status_code != 200:
            print('get fail',resp.status_code, URL)
        
            

        return resp

    def extract_top_250(self):
        """
        Extract the top 250 movies from the top 250 page and use them as seed for the crawler to start crawling.
        """
        # TODO update self.not_crawled and self.added_ids

        response = self.crawl(self.top_250_URL)
        
        if response:
            soup = BeautifulSoup(response.text, 'html.parser')

            movies = soup.find_all('a',{"class": "ipc-title-link-wrapper"})

            # print(len(movies))
            # print(type(movies))
            # print(movies)

            for movie in movies:
                movie_id = self.get_id_from_URL(movie['href'])
                if not movie_id.startswith('tt'):
                    continue
                movie_URL = f'https://www.imdb.com/title/{movie_id}/'
                
                
                if movie_id not in self.added_ids:
                    
                    self.not_crawled.append(movie_URL)
                    self.added_ids.add(movie_id)


    def get_imdb_instance(self):
        return {
            'id': None,  # str
            'title': None,  # str
            'first_page_summary': None,  # str
            'release_year': None,  # str
            'mpaa': None,  # str
            'budget': None,  # str
            'gross_worldwide': None,  # str
            'rating': None,  # str
            'directors': None,  # List[str]
            'writers': None,  # List[str]
            'stars': None,  # List[str]
            'related_links': None,  # List[str]
            'genres': None,  # List[str]
            'languages': None,  # List[str]
            'countries_of_origin': None,  # List[str]
            'summaries': None,  # List[str]
            'synopsis': None,  # List[str]
            'reviews': None,  # List[List[str]]
        }

    def start_crawling(self):
        """
        Start crawling the movies until the crawling threshold is reached.
        TODO: 
            replace WHILE_LOOP_CONSTRAINTS with the proper constraints for the while loop.
            replace NEW_URL with the new URL to crawl.
            replace THERE_IS_NOTHING_TO_CRAWL with the condition to check if there is nothing to crawl.
            delete help variables.

        ThreadPoolExecutor is used to make the crawler faster by using multiple threads to crawl the pages.
        You are free to use it or not. If used, not to forget safe access to the shared resources.
        """

        # help variables
        
        self.extract_top_250()
        
        futures = []
        crawled_counter = 0
        print('start thread')
        with ThreadPoolExecutor(max_workers=20) as executor:
            while crawled_counter <= self.crawling_threshold and self.not_crawled:
        
                with self.add_queue_lock:
                    URL = self.not_crawled.popleft()
        
                futures.append(executor.submit(self.crawl_page_info, URL))
                crawled_counter += 1
                if crawled_counter % 50 == 0:
                    print('counter:',crawled_counter)

                if len(self.not_crawled)==0:
                    wait(futures)
                    futures = []
                    

            wait(futures)
    def crawl_page_info(self, URL):
        """
        Main Logic of the crawler. It crawls the page and extracts the information of the movie.
        Use related links of a movie to crawl more movies.
        
        Parameters
        ----------
        URL: str
            The URL of the site
        """
        print("new iteration")
        
        response = self.crawl(URL)
  
        if response:
            parsed = BeautifulSoup(response.text, 'html.parser')

            movies = parsed.find_all('a',{"class": "ipc-poster-card__title ipc-poster-card__title--clamp-2 ipc-poster-card__title--clickable"})

            for movie in movies:
                movie_id = self.get_id_from_URL(movie['href'])
  
                if not movie_id.startswith('tt'):
                    continue
  
                movie_URL = f'https://www.imdb.com/title/{movie_id}/'
                if movie_id not in self.added_ids:
  
                    with self.add_queue_lock:
                        self.not_crawled.append(movie_URL)
  
                    with self.add_list_lock:
                        self.added_ids.add(movie_id)
  
        movie = self.get_imdb_instance()
        self.extract_movie_info(parsed ,movie, URL)
        with self.add_list_lock:
            self.crawled.append(movie)


    def extract_movie_info(self, res, movie, URL):
        """
        Extract the information of the movie from the response and save it in the movie instance.

        Parameters
        ----------
        res: requests.models.Response
            The response of the get request
        movie: dict
            The instance of the movie
        URL: str
            The URL of the site
        """
        movie['id'] = self.get_id_from_URL(URL)
        movie['title'] = self.get_title(res)
        movie['first_page_summary'] = self.get_first_page_summary(res)
        movie['release_year'] = self.get_release_year(res)
        movie['mpaa'] = self.get_mpaa(res)
        movie['budget'] = self.get_budget(res)
        movie['gross_worldwide'] = self.get_gross_worldwide(res)
        movie['directors'] = self.get_director(res)
        movie['writers'] = self.get_writers(res)
        movie['stars'] = self.get_stars(res)
        movie['related_links'] = self.get_related_links(res)
        movie['genres'] = self.get_genres(res)
        movie['languages'] = self.get_languages(res)
        movie['countries_of_origin'] = self.get_countries_of_origin(res)
        movie['rating'] = self.get_rating(res)
  
        summery_link = self.get_summary_link(URL)
        response = self.crawl(summery_link)
        parsed = BeautifulSoup(response.text, 'html.parser')
        
        movie['summaries'] = self.get_summary(parsed)
        movie['synopsis'] = self.get_synopsis(parsed)

        reveiw_link = self.get_review_link(URL)
        response = self.crawl(reveiw_link)
        parsed = BeautifulSoup(response.text, 'html.parser')
        movie['reviews'] = self.get_reviews_with_scores(parsed)


    def get_summary_link(self, url):
        """
        Get the link to the summary page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/plotsummary is the summary page

        Parameters
        ----------
        url: str
            The URL of the site
        Returns
        ----------
        str
            The URL of the summary page
        """
        try:
            movie_id = self.get_id_from_URL(url)
            return f'https://www.imdb.com/title/{movie_id}/plotsummary/'
        except:
            print("failed to get summary link")

    def get_review_link(self, url):
        """
        Get the link to the review page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/reviews is the review page
        """
        try:
            movie_id = self.get_id_from_URL(url)
            return f'https://www.imdb.com/title/{movie_id}/reviews/'
        except:
            print("failed to get review link")

    def get_title(self, soup):
        """
        Get the title of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The title of the movie

        """
        try:
            return soup.find('span',{"class": "hero__primary-text"}).get_text()
        except Exception as e:
            
            print("failed to get title")

    def get_first_page_summary(self, soup):
        """
        Get the first page summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The first page summary of the movie
        """
        try:
            return soup.find('span',{"data-testid": "plot-l"}).get_text()
        except:
            print("failed to get first page summary")

    def get_director(self, soup):
        """
        Get the directors of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The directors of the movie
        """
        try:
            all = soup.find(string='Directors')
            if all is None:
                all = soup.find(string='Director')
            
            all = all.parent.parent.find_all('a',{'class':"ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link"})
            return [i.get_text() for i in all]
        except:
            return []
            print("failed to get director")

    def get_stars(self, soup):
        """
        Get the stars of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The stars of the movie
        """
        try:
            all = soup.find(string='Stars')
            if all is None:
                all = soup.find(string='Star')
            
            all = all.parent.parent.find_all('a',{'class':"ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link"})
            return [i.get_text() for i in all]
        except:
            print("failed to get stars")

    def get_writers(self, soup):
        """
        Get the writers of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The writers of the movie
        """
        try:
            all = soup.find(string='Writers')
            if all is None:
                all = soup.find(string='Writer')

            all = all.parent.parent.find_all('a',{'class':"ipc-metadata-list-item__list-content-item ipc-metadata-list-item__list-content-item--link"})
            
            return [i.get_text() for i in all]
        except Exception as e:
            return []
            print("failed to get writers",e)

    def get_related_links(self, soup):
        """
        Get the related links of the movie from the More like this section of the page from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The related links of the movie
        """
        try:
            all = soup.find('section',{'data-testid':"MoreLikeThis"})
            all = all.find_all('a',{'class': "ipc-poster-card__title ipc-poster-card__title--clamp-2 ipc-poster-card__title--clickable"})
            return ['https://www.imdb.com/'+i['href'] for i in all]
        except Exception as e:
            print("failed to get related links",e)

    def get_summary(self, soup):
        """
        Get the summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The summary of the movie
        """
        try:
            all = soup.find('div',{'data-testid':"sub-section-summaries"})
            all = all.find_all('div',{'class': "ipc-html-content-inner-div"})
            return [i.get_text() for i in all]
        except:
            return []
            print("failed to get summary")

    def get_synopsis(self, soup):
        """
        Get the synopsis of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The synopsis of the movie
        """
        try:
            all = soup.find('div',{'data-testid':"sub-section-synopsis"})
            all = all.find_all('div',{'class': "ipc-html-content-inner-div"})
            return [i.get_text() for i in all]
        except:
            print("failed to get synopsis")
            return []

    def get_reviews_with_scores(self, soup):
        """
        Get the reviews of the movie from the soup
        reviews structure: [[review,score]]

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[List[str]]
            The reviews of the movie
        """
        try:

            all = soup.find('div',{'class':"lister-list"})
            ans = []
            
            for review in all.findChildren(recursive=False):
   
                # print(review)
                text = review.find('a', {'class':'title'}).get_text()
                # print('2')
                try:
                    score = review.find('span', {'class':'rating-other-user-rating'}).find('span').get_text()
                except:
                    score = ''
                ans.append([text,score])
            return ans
        except Exception as e:
            print("failed to get reviews", e)
            print(review)

    def get_genres(self, soup):
        """
        Get the genres of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The genres of the movie
        """
        try:
            all = soup.find_all('span', {'class': 'ipc-chip__text'})[:-1]
            return [i.contents[0] for i in all]
        except Exception as e:
            print("Failed to get generes",e)

    def get_rating(self, soup):
        """
        Get the rating of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The rating of the movie
        """
        try:
            
            return soup.find('span',{'class':"sc-bde20123-1 cMEQkK"}).get_text()
        except:
            return ''
            print("failed to get rating")

    def get_mpaa(self, soup):
        """
        Get the MPAA of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The MPAA of the movie
        """
        try:
            return soup.find('a',{"class": "ipc-link ipc-link--baseAlt ipc-link--inherit-color"},href=re.compile("parentalguide")).get_text()
        except:
            # print("failed to get mpaa")
            return('')

    def get_release_year(self, soup):
        """
        Get the release year of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The release year of the movie
        """
        try:
            return soup.find('a',{"class": "ipc-link ipc-link--baseAlt ipc-link--inherit-color"},href=re.compile("releaseinfo")).get_text()
                                    
        except:
            return ''
            print("failed to get release year")

    def get_languages(self, soup):
        """
        Get the languages of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The languages of the movie
        """
        try:
            all = soup.find_all('a',href=re.compile("primary_language"))
            return [i.get_text() for i in all]
            
        except:
            print("failed to get languages")
            return None

    def get_countries_of_origin(self, soup):
        """
        Get the countries of origin of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The countries of origin of the movie
        """
        try:
            all = soup.find_all('a',href=re.compile("country_of_origin="))
            return [i.get_text() for i in all]
        except:
            print("failed to get countries of origin")

    def get_budget(self, soup):
        """
        Get the budget of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The budget of the movie
        """
        try:
            # print(soup.find('div', {'data-testid':"title-boxoffice-section"}))
            return soup.find('div', {'data-testid':"title-boxoffice-section"}).find('span',{"class": "ipc-metadata-list-item__list-content-item"}).get_text()
        except Exception as e:
            return ''
            # print("failed to get budget",e)

    def get_gross_worldwide(self, soup):
        """
        Get the gross worldwide of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The gross worldwide of the movie
        """
        try:
            return soup.find('li', {'data-testid':"title-boxoffice-cumulativeworldwidegross"}).find('span',{"class": "ipc-metadata-list-item__list-content-item"}).get_text()
        except:
            return ''
            #print("failed to get gross worldwide")


def main():
    imdb_crawler = IMDbCrawler(crawling_threshold=1050)
    # imdb_crawler.read_from_file_as_json()
    imdb_crawler.start_crawling()
    print('done. writing...')
    imdb_crawler.write_to_file_as_json()


if __name__ == '__main__':
    main()
