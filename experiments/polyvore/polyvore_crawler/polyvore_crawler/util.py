import json
import os
import re

import boto

from boto.s3.key import Key
from smart_open import smart_open, s3_iter_bucket
from tqdm import tqdm


def clean_url(url, url_root='http://www.polyvore.com'):
    if url.startswith('../'):
        url = url[3:]
    if not url.startswith('http://') and not url.startswith('https://'):
        url = os.path.join(url_root, url)
    return url


def load_local_dir_jls(path):
    for name in os.listdir(path):
        if name.endswith('.jl'):
            with open(os.path.join(path, name)) as infile:
                for line in infile:
                    if line:
                        yield json.loads(line)


def load_dir_jls(conn, path):
    bucket, key = get_bucket_key(path)

    bucket = conn.get_bucket(bucket)
    for name, content in tqdm(
            s3_iter_bucket(
                bucket, key, accept_key=lambda name: name.endswith('.jl'))):
        for line in content.decode('utf8').split('\n'):
            if line:
                yield json.loads(line)


def load_jl(path):
    with smart_open(path) as infile:
        for line in infile:
            yield json.loads(line.decode('utf8'))


def load_local_jl(path):
    with open(path) as infile:
        for line in infile:
            yield json.loads(line)


def dump_jl(items):
    return '\n'.join([json.dumps(item) for item in items]).encode('utf8')


def connect_s3(aws_access_key_id, aws_secret_access_key):
    return boto.connect_s3(aws_access_key_id, aws_secret_access_key)


def get_bucket_key(path):
    _, _, bucket, key = path.split('/', 3)
    return bucket, key


def s3_set_file_contents(conn, bucket, key, s):
    b = conn.get_bucket(bucket)
    k = Key(b)
    k.key = key
    return k.set_contents_from_string(s)


def list_dir(conn, path, show_dir=False):
    m = re.match(r'(s3n?)://([^/]+)/(.+)', path)
    if not m:
        raise ValueError("invalid s3 path %s" % path)
    prefix, bucket_name, path = m.groups()
    bucket = conn.get_bucket(bucket_name, validate=False)
    files = []
    visited = set()
    for item in bucket.list(path):
        item = item.key
        if item.startswith(path):
            subpath = item[len(path):].lstrip('/')
            if show_dir:
                if '/' in subpath:
                    subpath = subpath.split('/')[0]
                else:
                    subpath = ''

            if (len(subpath) > 0 and not subpath.startswith('_') and
                    '/' not in subpath and not subpath.endswith('$')):
                file_path = prefix + '://' + os.path.join(bucket_name, path,
                                                          subpath)
                if file_path not in visited:
                    visited.add(file_path)
                    files.append(file_path)
    return files


def produce_s3_url(path, aws_access_key_id, aws_secret_access_key):
    if (aws_access_key_id and aws_secret_access_key and
            path.startswith('s3://')):
        bucket, key = get_bucket_key(path)
        if '@' not in bucket:
            path = 's3://{}:{}@{}/{}'.format(
                aws_access_key_id, aws_secret_access_key, bucket, key)

    return path
