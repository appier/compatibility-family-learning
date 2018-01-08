import json

import boto

from boto.s3.key import Key
from smart_open import s3_iter_bucket
from tqdm import tqdm


def set_file_contents(conn, path, s):
    bucket, key = get_bucket_key(path)
    b = conn.get_bucket(bucket, validate=False)
    k = Key(b)
    k.key = key
    return k.set_contents_from_string(s)


def get_file_contents(conn, path):
    bucket, key = get_bucket_key(path)
    b = conn.get_bucket(bucket, validate=False)
    k = Key(b)
    k.key = key
    return k.get_contents_as_string()


def file_exists(conn, path):
    bucket, key = get_bucket_key(path)

    bucket = conn.get_bucket(bucket, validate=False)
    return bucket.get_key(key) is not None


def copy_file(conn, src, dst):
    src_bucket, src_key = get_bucket_key(src)
    dst_bucket, dst_key = get_bucket_key(dst)
    bucket = conn.get_bucket(dst_bucket, validate=False)
    bucket.copy_key(
        new_key_name=dst_key, src_bucket_name=src_bucket, src_key_name=src_key)


def connect_s3():
    return boto.connect_s3()


def get_bucket_key(path):
    _, _, bucket, key = path.split('/', 3)
    return bucket, key


def load_dir_jls(conn, path):
    bucket, key = get_bucket_key(path)

    bucket = conn.get_bucket(bucket, validate=False)
    for name, content in tqdm(
            s3_iter_bucket(
                bucket, key, accept_key=lambda name: name.endswith('.jl'))):
        for line in content.decode('utf8').split('\n'):
            if line:
                yield json.loads(line)
