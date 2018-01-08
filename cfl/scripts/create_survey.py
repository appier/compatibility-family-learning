"""
Create a survey from different methods
"""

import argparse
import hashlib
import random
import shutil
import os

from scipy.misc import imread, imsave


def parse_disco(path, output_dir, names):
    sources = {}
    for name in os.listdir(path):
        if 'ids.txt' in name:
            prefix = name.split('_')[0]
            id_path = os.path.join(path, name)
            with open(id_path) as infile:
                for i, asin in enumerate(infile):
                    asin = asin.strip()
                    sources[asin] = (prefix, i)

    output_disco_dir = os.path.join(output_dir, 'disco')
    for i, name in enumerate(names):
        prefix, seq = sources[name]
        image_path = os.path.join(path, '{}_x_AB.png'.format(prefix))
        j = seq // 8
        k = seq % 8
        image = imread(image_path)[2 + j * 66:2 + j * 66 + 64, 2 + k * 66:
                                   2 + k * 66 + 64, :]
        imsave(
            os.path.join(output_disco_dir, '{}_{}.png'.format(i, name)), image)


def parse_mrcgan(path, disco_path, output_dir, num_outputs):
    output_mrcgan_dir = os.path.join(output_dir, 'mrcgan')
    source_dir = os.path.join(output_dir, 'source')

    disco_names = set()
    for name in os.listdir(disco_path):
        if '.txt' in name:
            with open(os.path.join(disco_path, name)) as infile:
                for line in infile:
                    disco_names.add(line.strip())
    names = sorted(
        {name.split('.')[0].split('_')[1]
         for name in os.listdir(path)})
    names = [
        name for name in names if name != 'B007U8EC8Q' and name in disco_names
    ]
    random.shuffle(names)
    names = names[:num_outputs]
    for i, name in enumerate(names):
        c = random.randint(0, 99)
        shutil.copyfile(
            os.path.join(path, 'test_' + name + '_src.png'),
            os.path.join(source_dir, '{}_{}.png'.format(i, name)))
        j = c % 10
        k = c // 10
        image = imread(os.path.join(path, 'test_' + name + '.png'))[
            j * 64:j * 64 + 64, k * 64:k * 64 + 64, :]
        imsave(
            os.path.join(output_mrcgan_dir, '{}_{}.png'.format(i, name)),
            image)

    return names


def parse_pix2pix(path, output_dir, names):
    files = {}
    for name in os.listdir(path):
        if '_fake_B' in name:
            files[name.split('_')[0]] = os.path.join(path, name)

    output_pix2pix_dir = os.path.join(output_dir, 'pix2pix')
    for i, name in enumerate(names):
        shutil.copyfile(files[name],
                        os.path.join(output_pix2pix_dir, '{}_{}.png'.format(
                            i, name)))


def create(root_dir, output_dir, num_outputs, seed):
    paths = [
        os.path.join(output_dir, 'mrcgan'),
        os.path.join(output_dir, 'pix2pix'),
        os.path.join(output_dir, 'disco'),
        os.path.join(output_dir, 'images'),
        os.path.join(output_dir, 'source')
    ]
    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    random.seed(seed)
    mrcgan_path = os.path.join(root_dir, 'mrcgan')
    disco_path = os.path.join(root_dir, 'disco')
    names = parse_mrcgan(mrcgan_path, disco_path, output_dir, num_outputs)
    pix2pix_path = os.path.join(root_dir, 'pix2pix')
    parse_pix2pix(pix2pix_path, output_dir, names)
    parse_disco(disco_path, output_dir, names)

    image_hs = []
    for model in ('mrcgan', 'disco', 'pix2pix'):
        path = os.path.join(output_dir, model)
        id_path = os.path.join(output_dir, model + '_ids.txt')
        with open(id_path, 'w') as id_file:
            hs = []
            for i, name in enumerate(names):
                image_path = os.path.join(output_dir, model,
                                          '{}_{}.png'.format(i, name))
                with open(image_path, 'rb') as image_file:
                    h = hashlib.md5(image_file.read()).hexdigest()
                output_path = os.path.join(output_dir, 'images', h + '.png')
                assert not os.path.exists(output_path)
                shutil.copyfile(image_path, output_path)
                id_file.write(h + '\n')
                hs.append(h)
        image_hs.append(hs)
    q_path = os.path.join(output_dir, 'qs.tsv')
    with open(q_path, 'w') as q_file:
        q_file.write('number\tsource\tA\tB\tC\n')
        for i, name in enumerate(names):
            out_tokens = [hs[i] for hs in image_hs]
            random.shuffle(out_tokens)
            outline = '\t'.join(out_tokens)
            q_file.write('{}\t{}\t{}\n'.format(i, name, outline))


def main():
    args = parse_args()
    create(**vars(args))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-outputs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1126)
    return parser.parse_args()


if __name__ == '__main__':
    main()
