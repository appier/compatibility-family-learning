# -*- coding: utf-8 -*-
import time

import scrapy

from scrapy.http import Request

from amazon_crawler.items import AmazonItem


class AmazonItemSpider(scrapy.Spider):
    name = 'amazon_item'

    def load_input(self, path):
        with open(path) as infile:
            for line in infile:
                line = line.strip()
                if line:
                    yield line.split('\t')

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = cls(*args, crawler=crawler, **kwargs)
        spider._set_crawler(crawler)
        return spider

    def __init__(self, crawler, input_path, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.aws_access_key_id = crawler.settings.get('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = crawler.settings.get(
            'AWS_SECRET_ACCESS_KEY')

        self.input_path = input_path

    def start_requests(self):
        # check if we can connect to Amazon.com
        yield Request('http://www.amazon.com', callback=self.parse)

    def parse(self, response):
        for asin, url in self.load_input(self.input_path):
            time.sleep(0.5)
            yield AmazonItem(asin=asin, image_urls=[url])
