import concurrent.futures

import json
import requests
from PIL import Image
from bs4 import BeautifulSoup
from tqdm import tqdm


class Scraper:

    def get_asset_urls(self, urls, out_file=None):
        """
        Retrieves URLs to raw images, given list of subpages containing comicA strips.
        Uses a single thread.
        :param urls: URLs to the subpages.
        :param out_file: File to save the output.
        :return: A list of URLs.
        """

        results = self.scrape_single_threaded(urls, self.get_asset_url)

        if out_file:
            json.dump(results, open(out_file, "w+"))

        return results

    def get_asset_urls_multithreaded(self, urls, out_file=None, max_workers=10):
        """
        Retrieves URLs to raw images, given list of subpages containing comic strips.
        Uses multiple threads.
        :param urls: URLs to the subpages.
        :param out_file: File to save the output.
        :param max_workers: Number of threads.
        :return: A list of URLs.
        """

        results = self.scrape_multi_threaded(urls, self.get_asset_url, max_workers)

        if out_file:
            json.dump(results, open(out_file, "w+"))

        return results

    def scrape_images(self, urls):
        """
        Scrapes images from urls. Returns raw image arrays. Uses a single thread.
        :param urls: URLs to scrape
        :return: List of raw image arrays.
        """

        results = self.scrape_single_threaded(urls, self.get_image_from_url)
        return results

    def scrape_images_multithreaded(self, urls, max_workers=10):
        """
        Scrapes images from urls. Returns raw image arrays. Uses multiple threads.
        :param urls: URLs to scrape
        :return: List of raw image arrays.
        """

        results = self.scrape_multi_threaded(urls, self.get_image_from_url, max_workers)
        return results

    @staticmethod
    def scrape_single_threaded(urls, handler):
        """
        Performs single threaded scraping.
        :param urls: URLs to scrape.
        :param handler: Function that handles sending the request and parsing the response.
        :return: Results of calling the handler on all URLs.
        """
        results = []
        for url in urls:
            results.append(handler(url))

        return results

    def scrape_and_save_multithreaded(self, urls, paths, max_workers):
        """
        Performs multi threaded scraping and saves images to given paths.
        :param urls: URLs to scrape.
        :param paths: Paths where the results should be saved.
        :param max_workers: Number of threads
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = list()
            for url, path in tqdm(zip(urls, paths)):
                futures.append(executor.submit(self.save_image_from_url, conf=(path, url)))

            results = list()
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

    def scrape_and_save_singlethreaded(self, urls, paths):
        """
        Performs single threaded scraping and saves images to given paths.
        :param urls: URLs to scrape.
        :param paths: Paths where the results should be saved.
        """
        for url, path in tqdm(zip(urls, paths)):
            self.save_image_from_url((path, url))

    def save_image_from_url(self, conf):
        path, url = conf
        im = self.get_image_from_url(url)
        if im:
            im.save(path)


    @staticmethod
    def scrape_multi_threaded(urls, handler, max_workers):
        """
        Performs multi threaded scraping.
        :param urls: URLs to scrape.
        :param handler: Function that handles sending the request and parsing the response.
        :param max_workers: Number of threads
        :return: Results of calling the handler on all URLs.
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = list()
            for url in urls:
                futures.append(executor.submit(handler, url=url))

            results = list()
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

            return results

    @staticmethod
    def get_asset_url(url):
        """
        Retrieves image asset URL by using the 'twitter:image' tag.
        :param url: URL to a page containing the image.
        :return: URL to the image asset.
        """
        # try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36'}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            comic = soup.find('meta', {'name': 'twitter:image'})
            image_url = comic["content"]
            return image_url
        else:
            return None

    @staticmethod
    def get_image_from_url(url):
        """
        Retrieves an image from URL.
        :param url: Link to the image.
        :return: Image array.
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36'}
            im = Image.open(requests.get(url, headers=headers, stream=True).raw)
            return im
        except:
            print(f"ERROR: Unable to retrieve image from URL: {url}.")