# Author scraper for public-domain-poetry.com
# Paste in an author's page URL (e.g. http://public-domain-poetry.com/emily-elizabeth-dickinson) as
# the argument and the script will download every poem from the linked subpages into download_cache/
# The site's robot.txt politely asks crawlers to wait 10 seconds between requests, so this script
# politely respects that and sleeps for 10 seconds between subpage downloads.

import argparse
import bs4
import os
import re
import time
import urllib.request

parser = argparse.ArgumentParser(prog = "Poetry Scraper")
parser.add_argument('url')
args = parser.parse_args()

url = args.url

AUTHOR_REGEX = re.compile(r'^http://(www\.)?public-domain-poetry\.com/([^/]*)$')
RELATIVE_PATH_REGEX = re.compile('^([^/]*)/([^/]*)$')
PAGE_TITLE_REGEX = re.compile('^Public Domain Poetry - (.*) by (.*)$')

match = AUTHOR_REGEX.match(url)

if not match:
    print(f"author not matched!")
    exit()

author = match[2]
print(f"scraping {author}...")
author_dirpath = os.path.join("download_cache", author)
os.makedirs(author_dirpath, exist_ok = True)

need_delay = False

author_index_filepath = os.path.join(author_dirpath, "author_index.html")
if os.path.isfile(author_index_filepath):
    print("using cached author_index.html")
    with open(author_index_filepath, "r") as f:
        author_index_contents = f.read()
    author_index_soup = bs4.BeautifulSoup(author_index_contents, features = "html.parser")
else:
    print("downloading author_index.html")
    author_index_contents = urllib.request.urlopen(url).read()
    author_index_soup = bs4.BeautifulSoup(author_index_contents, features = "html.parser")
    with open(author_index_filepath, "w", encoding = "utf-8") as f:
        f.write(author_index_soup.prettify())
    need_delay = True

poem_urls = dict()
for link in author_index_soup.find_all('a'):
    relative_path = link.get('href')
    match = RELATIVE_PATH_REGEX.match(relative_path)
    if not match or match[1] != author:
        continue
    poem_urls[match[2]] = "http://www.public-domain-poetry.com/" + relative_path

for poem_filename, poem_url in poem_urls.items():
    poem_filepath = os.path.join(author_dirpath, poem_filename + ".txt")
    if not os.path.isfile(poem_filepath):
        if need_delay:
            print("sleeping...")
            time.sleep(10)
        print(f"downloading {poem_url}")
        poem_contents = urllib.request.urlopen(poem_url).read()
        need_delay = True
        poem_soup = bs4.BeautifulSoup(poem_contents, features = "html.parser")
        page_title = poem_soup.title.get_text()
        match = PAGE_TITLE_REGEX.match(page_title)
        if not match:
            print(f"Page title did not match regex! Was: {page_title}")
            continue
        file_contents = f"Title: {match[1]}\nAuthor: {match[2]}\n\n*** START TEXT ***\n\n"
        for tag in poem_soup.find_all(class_ = "t3a"):
            tag_lines = tag.get_text().splitlines()
            for line in tag_lines:
                file_contents += line.strip() + "\n"
        file_contents += "\n*** END TEXT ***\n"
        with open(poem_filepath, "w", encoding = "utf-8") as f:
            f.write(file_contents)
