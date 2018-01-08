# -*- coding: utf-8 -*-

# Scrapy settings for polyvore_crawler project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     http://doc.scrapy.org/en/latest/topics/settings.html
#     http://scrapy.readthedocs.org/en/latest/topics/downloader-middleware.html
#     http://scrapy.readthedocs.org/en/latest/topics/spider-middleware.html

BOT_NAME = 'polyvore_crawler'

SPIDER_MODULES = ['polyvore_crawler.spiders']
NEWSPIDER_MODULE = 'polyvore_crawler.spiders'

# Obey robots.txt rules
ROBOTSTXT_OBEY = True

# The S3 path to store items
#ITEMS_STORE = ...
# The S3 path to store images
#IMAGES_STORE = ...
# AWS S3 Keys
#AWS_ACCESS_KEY_ID = ...
#AWS_SECRET_ACCESS_KEY = ...
