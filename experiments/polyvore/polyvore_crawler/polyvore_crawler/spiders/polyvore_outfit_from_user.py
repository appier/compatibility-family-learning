# -*- coding: utf-8 -*-
import json
import os

from urllib.parse import urlencode

import scrapy

from scrapy.http import Request
from scrapy.selector import Selector

from polyvore_crawler.items import OutfitItem


class PolyvoreOutfitFromUserSpider(scrapy.Spider):
    name = 'polyvore_outfit_from_user'
    allowed_domains = ['polyvore.com']
    query_url_format = 'http://{}.polyvore.com/cgi/profile'
    author_url_format = 'http://{}.polyvore.com'

    custom_settings = {'CONCURRENT_REQUESTS': 1, 'RETRY_HTTP_CODES': []}
    handle_httpstatus_list = [403]

    def load_input(self, path):
        with open(path) as infile:
            return {json.loads(l)['author'] for l in infile}

    def load_output(self, path):
        with open(path) as infile:
            return {json.loads(l)['url'] for l in infile}

    def load_author(self, path):
        with open(path) as infile:
            return {l.strip() for l in infile}

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = cls(*args, crawler=crawler, **kwargs)
        spider._set_crawler(crawler)
        return spider

    def __init__(self, crawler, input_path, output_path, author_path, *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if os.path.exists(output_path):
            self.visited = self.load_output(output_path)
        else:
            self.visited = set()

        if os.path.exists(author_path):
            self.visited_author = self.load_author(author_path)
        else:
            self.visited_author = set()
        self.author_path = author_path

        self.authors = self.load_input(input_path)

        crawler.stats.set_value('authors', len(self.authors))

    def construct_url(self, author, uid, page, passback=None):
        params = {'.in': 'json', '.out': 'jsonx'}
        req = {'page': page, 'filter': 'sets', '.in': 'json', 'id': uid}
        if passback is not None:
            req['.passback'] = passback

        params['request'] = json.dumps(req)

        return PolyvoreOutfitFromUserSpider.query_url_format.format(
            author) + '?' + urlencode(params)

    def start_requests(self):
        for author in self.authors:
            if author not in self.visited_author:
                url = PolyvoreOutfitFromUserSpider.author_url_format.format(
                    author)
                yield Request(
                    url, callback=self.parse, meta={'author': author})

    def parse_api(self, response):
        author = response.meta['author']

        if response.status == 403:
            with open(self.author_path, 'a') as outfile:
                outfile.write('{}\n'.format(author))
            self.visited_author.add(author)
            return

        result = json.loads(response.body_as_unicode())
        selector = Selector(text=result['result']['html'])
        for item in selector.css('div.grid_item'):
            author = item.xpath(
                'div[@class="under"]/div[@class="createdby"]//a/text()'
            ).extract_first()
            title = item.xpath(
                'div[@class="under"]/div[@class="title"]//a/text()'
            ).extract_first()
            url = item.xpath('div[@class="main"]/a/@href').extract_first()
            fav_count = item.xpath(
                'div[@class="under"]/div[@class="container"]//span[@class="fav_count"]/text()'
            ).extract_first()
            if url not in self.visited:
                yield OutfitItem(
                    url=url, title=title, author=author, fav_count=fav_count)
                self.visited.add(url)

        if result['result']['more_pages'] == 1:
            passback = result['.passback']
            page = response.meta['page'] + 1
            uid = response.meta['uid']
            url = self.construct_url(author, uid, page, passback)
            yield Request(
                url,
                callback=self.parse_api,
                meta={'uid': uid,
                      'page': page,
                      'author': author})
        else:
            with open(self.author_path, 'a') as outfile:
                outfile.write('{}\n'.format(author))
            self.visited_author.add(author)

    def parse(self, response):
        if response.status == 200:
            uid = response.xpath(
                '//input[@name=".done"]/@value').extract_first()
            if uid is not None:
                uid = uid.split('=')[-1]
                author = response.meta['author']
                page = 1
                url = self.construct_url(author, uid, page)
                yield Request(
                    url,
                    callback=self.parse_api,
                    meta={'uid': uid,
                          'page': page,
                          'author': author})
