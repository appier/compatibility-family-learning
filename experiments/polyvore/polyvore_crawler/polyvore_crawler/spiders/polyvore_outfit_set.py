# -*- coding: utf-8 -*-
import json
import os

import scrapy

from scrapy.http import Request
from smart_open import smart_open

from polyvore_crawler.items import OutfitSetItem
from polyvore_crawler.util import produce_s3_url


class PolyvoreOutfitSetSpider(scrapy.Spider):
    name = 'polyvore_outfit_set'
    allowed_domains = ['www.polyvore.com']
    url_root = 'http://www.polyvore.com'

    def clean_url(self, url):
        if url.startswith('../'):
            url = url[3:]
        if not url.startswith('http://') and not url.startswith('https://'):
            url = os.path.join(PolyvoreOutfitSetSpider.url_root, url)
        return url

    def load_input(self, path):
        with smart_open(path) as infile:
            return [json.loads(l.decode('utf8')) for l in infile]

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = cls(*args, crawler=crawler, **kwargs)
        spider._set_crawler(crawler)
        return spider

    def __init__(self, crawler, input_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visited = set()

        aws_access_key_id = crawler.settings.get('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = crawler.settings.get('AWS_SECRET_ACCESS_KEY')

        input_path = produce_s3_url(input_path, aws_access_key_id,
                                    aws_secret_access_key)

        self.seeds = self.load_input(input_path)

    def start_requests(self):
        for item in self.seeds:
            url = self.clean_url(item['url'])
            if url not in self.visited:
                yield Request(url, callback=self.parse)
                self.visited.add(url)

    def parse(self, response):
        title = response.xpath(
            '//div[@id="set_title"]//h1/text()').extract_first()
        desc = response.css('div.actual_set_description').extract_first()
        fav_count = response.xpath(
            '//span[@class="fav_count"]/text()').extract_first()
        author = response.xpath('//a[@rel="author"]/text()').extract_first()
        items = response.xpath(
            '//ul[@trackcontext="set_items"]//a[@trackcontext="image"]/@href'
        ).extract()
        yield OutfitSetItem(
            title=title,
            desc=desc,
            fav_count=fav_count,
            author=author,
            items=items,
            url=response.url)
