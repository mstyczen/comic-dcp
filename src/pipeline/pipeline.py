import datetime
import cv2
from scraper.Scraper import Scraper
from panel_extractor.PanelExtractor import PanelExtractor
from text_extractor.TextExtractor import extract_text
from utils import get_dates_list

if __name__ == "__main__":
    # get comic urls
    start_date = datetime.datetime(2021, 1, 1, 0, 0)
    end_date = datetime.datetime(2021, 1, 10, 0, 0)
    dates = get_dates_list(start_date, end_date)
    date_strings = [str(date.strftime('%Y-%m-%d')) for date in dates]
    comic_urls = [f"http://dilbert.com/strip/{date_string}" for date_string in date_strings]

    scraper = Scraper()
    # get asset url from the main page
    urls = scraper.get_asset_urls_multithreaded(comic_urls)
    # determing output paths, scrape images and save them
    full_images_directory = "../../data/full-images"
    full_images_paths = [f"{full_images_directory}/{date}.png" for date in date_strings]
    scraper.scrape_and_save_multithreaded(urls, full_images_paths, max_workers=5)

    # extract panels from each full illustration
    splitted_images_directory = "../../data/splitted-images"
    panel_extractor = PanelExtractor()
    splitted_images_paths = panel_extractor.extract_and_save_panels(full_images_paths, splitted_images_directory)

    # get transcriptions
    transcriptions = extract_text(splitted_images_paths, "vision-api", rescale=True, clustering=True)

    # visualize
    for path in splitted_images_paths:
        image = cv2.imread(path)
        cv2.imshow("path", image)
        print(transcriptions[path])
        cv2.waitKey(0)
