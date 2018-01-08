import json
import os
import re

import boto

from boto.s3.key import Key
from smart_open import s3_iter_bucket
from tqdm import tqdm


def load_dir_jls(conn, path):
    bucket, key = get_bucket_key(path)

    bucket = conn.get_bucket(bucket, validate=False)
    for name, content in tqdm(
            s3_iter_bucket(
                bucket, key, accept_key=lambda name: name.endswith('.jl'))):
        for line in content.decode('utf8').split('\n'):
            if line:
                yield json.loads(line)


def connect_s3(aws_access_key_id, aws_secret_access_key):
    return boto.connect_s3(aws_access_key_id, aws_secret_access_key)


def get_bucket_key(path):
    _, _, bucket, key = path.split('/', 3)
    return bucket, key


def s3_set_file_contents(conn, bucket, key, s):
    b = conn.get_bucket(bucket, validate=False)
    k = Key(b)
    k.key = key
    return k.set_contents_from_string(s)


def s3_get_file_contents(conn, bucket, key):
    b = conn.get_bucket(bucket, validate=False)
    k = Key(b)
    k.key = key
    return k.get_contents_as_string()


def dump_jl(items):
    return '\n'.join([json.dumps(item) for item in items]).encode('utf8')
