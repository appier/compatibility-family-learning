# -*- coding: utf-8 -*-
import json
import os

from urllib.parse import urlparse, parse_qs

import scrapy

from scrapy.http import Request
from smart_open import smart_open

import polyvore_crawler.categories

from polyvore_crawler.items import ProductItem
from polyvore_crawler.util import produce_s3_url


class PolyvoreItemSpider(scrapy.Spider):
    name = 'polyvore_item'
    allowed_domains = ['www.polyvore.com']
    url_root = 'http://www.polyvore.com'
    handle_httpstatus_list = [404]

    def clean_url(self, url):
        if url.startswith('../'):
            url = url[3:]
        if not url.startswith('http://') and not url.startswith('https://'):
            url = os.path.join(PolyvoreItemSpider.url_root, url)
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
            try:
                parsed_url = urlparse(url)
                name = parsed_url.path.split('/')[-2]
                product_id = parse_qs(parsed_url.query)['id'][0]
                tag = (name, product_id)
                if tag not in self.visited:
                    yield Request(url, callback=self.parse)
                    self.visited.add(tag)
            except Exception as e:
                self.logger.warning('Exception: %s', e)

    def parse(self, response):
        parsed_url = urlparse(response.url)
        name = parsed_url.path.split('/')[-2]
        product_id = parse_qs(parsed_url.query)['id'][0]

        if response.status == 404:
            yield ProductItem(
                url=response.url,
                name=name,
                product_id=product_id,
                image_urls=[])
        else:
            categories = response.xpath(
                '//div[contains(@class, "breadcrumb")]//span[@itemprop="title"]/text()'
            ).extract()
            title = response.xpath('//h1/text()').extract_first()
            price = response.css('span.price').xpath('text()').extract_first()
            desc = response.xpath(
                '//div[@class="tease"]/text()').extract_first()
            paid_url = response.css(
                'div.buy_button a::attr(href)').extract_first()

            if polyvore_crawler.categories.belongs_to(
                    categories, polyvore_crawler.categories.all_categories):
                image_urls = response.xpath(
                    '//center[@id="thing_img"]//a/img/@src').extract()
            else:
                image_urls = []

            yield ProductItem(
                url=response.url,
                name=name,
                product_id=product_id,
                image_urls=image_urls,
                title=title,
                categories=categories,
                price=price,
                desc=desc,
                paid_url=paid_url)
