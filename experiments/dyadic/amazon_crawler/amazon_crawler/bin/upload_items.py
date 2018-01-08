import argparse
import json
import os

from scrapy.utils.project import get_project_settings

from ..util import connect_s3, s3_set_file_contents, get_bucket_key


def parse_args():
    settings = get_project_settings()
    dash_settings = json.loads(os.environ.get('SHUB_SETTINGS', '{}'))
    settings.setdict(
        dash_settings.get('project_settings', {}), priority='project')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--aws-access-key-id', default=settings['AWS_ACCESS_KEY_ID'])
    parser.add_argument(
        '--aws-secret-access-key', default=settings['AWS_SECRET_ACCESS_KEY'])
    parser.add_argument('--items-store', default=settings['ITEMS_STORE'])
    parser.add_argument('--spider-name', default='amazon_item')
    parser.add_argument('--input-files', nargs='+', required=True)
    return parser.parse_args()


def upload_items(aws_access_key_id, aws_secret_access_key, items_store,
                 spider_name, input_files):
    conn = connect_s3(aws_access_key_id, aws_secret_access_key)
    for path in input_files:
        root = os.path.join(items_store, spider_name)
        bucket, prefix = get_bucket_key(root)

        with open(path) as infile:
            items = infile.read()
            file_name = os.path.join(prefix, os.path.basename(path) + '.jl')
            s3_set_file_contents(conn, bucket, file_name, items)


def main():
    args = parse_args()
    upload_items(**vars(args))


if __name__ == '__main__':
    main()
