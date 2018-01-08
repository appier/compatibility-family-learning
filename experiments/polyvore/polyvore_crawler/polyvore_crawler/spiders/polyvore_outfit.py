# -*- coding: utf-8 -*-
import json
import os

from urllib.parse import urlencode

import scrapy

from scrapy.http import Request
from scrapy.selector import Selector

from polyvore_crawler.items import OutfitItem


class PolyvoreOutfitSpider(scrapy.Spider):
    name = 'polyvore_outfit'
    allowed_domains = ['www.polyvore.com']
    query_url = 'http://www.polyvore.com/cgi/search.sets'
    colors = (
        None,
        'black',
        'white',
        'grey',
        'brown',
        'beige',
        'red',
        'pink',
        'orange',
        'yellow',
        'blue',
        'light blue',
        'green',
        'purple', )
    dates = (
        None,
        'day',
        'week',
        'month',
        '3m', )
    categories = ('fashion', )
    queries = (
        None,
        'street style',
        'celebrity',
        'work wear',
        'formal',
        'vacation',
        'school',
        'edgy',
        'casual',
        'preppy',
        'boho',
        'sporty',
        'plus size',
        'style',
        'cute',
        'good',
        'love',
        'happy',
        'pretty',
        'best',
        'teenage', )

    def load_output(self, path):
        with open(path) as infile:
            return {json.loads(l)['url'] for l in infile}

    def __init__(self, output_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if os.path.exists(output_path):
            self.visited = self.load_output(output_path)
        else:
            self.visited = set()
        self.url = set()

    def construct_url(self,
                      limit=30,
                      start=0,
                      category=None,
                      color=None,
                      date=None,
                      query=None):
        params = {'.in': 'json', '.out': 'jsonx'}
        req = {'.passback': {'next_token': {'limit': limit, 'start': start}}}

        if category is not None:
            params['category'] = category
        if date is not None:
            params['date'] = date
        if color is not None:
            params['color'] = color
        if query is not None:
            params['query'] = query

        params['request'] = json.dumps(req)

        return PolyvoreOutfitSpider.query_url + '?' + urlencode(params)

    def start_requests(self):
        for query in PolyvoreOutfitSpider.queries:
            for date in PolyvoreOutfitSpider.dates:
                for category in PolyvoreOutfitSpider.categories:
                    for color in PolyvoreOutfitSpider.colors:
                        url = self.construct_url(
                            date=date,
                            category=category,
                            color=color,
                            query=query)
                        if url in self.url:
                            print(url, 'is the same !!')
                            raise Exception()
                        self.url.add(url)
                        yield Request(url, callback=self.parse)

    def parse(self, response):
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
            if url not in self.visited:
                yield OutfitItem(url=url, title=title, author=author)
                self.visited.add(url)

        if result['result']['more_pages'] == 1:
            token = result['.passback']['next_token']
            limit = token['limit']
            start = token['start']
            url = self.construct_url(limit=limit, start=start)
            yield Request(url, callback=self.parse)
