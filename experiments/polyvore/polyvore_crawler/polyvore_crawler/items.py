# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class OutfitItem(scrapy.Item):
    title = scrapy.Field()
    author = scrapy.Field()
    url = scrapy.Field()
    fav_count = scrapy.Field()


class OutfitSetItem(scrapy.Item):
    title = scrapy.Field()
    author = scrapy.Field()
    fav_count = scrapy.Field()
    url = scrapy.Field()
    items = scrapy.Field()
    desc = scrapy.Field()


class ProductItem(scrapy.Item):
    name = scrapy.Field()
    product_id = scrapy.Field()
    url = scrapy.Field()
    images = scrapy.Field()
    image_urls = scrapy.Field()
    title = scrapy.Field()
    categories = scrapy.Field()
    price = scrapy.Field()
    desc = scrapy.Field()
    paid_url = scrapy.Field()
