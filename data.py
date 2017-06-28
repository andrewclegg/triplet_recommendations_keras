from __future__ import print_function

import itertools
import os
import zipfile
import shutil

import numpy as np

import requests

import scipy.sparse as sp


def _get_movielens_path():
    """
    Get path to the movielens dataset file.
    """

    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'movielens.zip')


def _download_movielens(dest_path):
    """
    Download the dataset.
    """

    url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
    req = requests.get(url, stream=True)

    print('Downloading MovieLens data')

    with open(dest_path, 'wb') as fd:
        for chunk in req.iter_content():
            fd.write(chunk)


def _get_raw_movielens_data():
    """
    Return the raw lines of the train and test files.
    """

    path = _get_movielens_path()

    if not os.path.isfile(path):
        _download_movielens(path)

    with zipfile.ZipFile(path) as datafile:
        return (datafile.read('ml-100k/ua.base').decode().split('\n'),
                datafile.read('ml-100k/ua.test').decode().split('\n'))


def _parse(data):
    """
    Parse movielens dataset lines.
    """

    for line in data:

        if not line:
            continue

        uid, iid, rating, timestamp = [int(x) for x in line.split('\t')]

        yield uid, iid, rating, timestamp


def _build_interaction_matrix(rows, cols, data):

    mat = sp.lil_matrix((rows, cols), dtype=np.int32)

    for uid, iid, rating, timestamp in data:
        # Let's assume only really good things are positives
        if rating >= 4.0:
            mat[uid, iid] = 1.0

    return mat.tocoo()


def _get_movie_raw_metadata():
    """
    Get raw lines of the movies file.
    """

    path = _get_movielens_path()

    if not os.path.isfile(path):
        _download_movielens(path)

    with zipfile.ZipFile(path) as datafile:
        return datafile.read('ml-100k/u.item').decode(errors='ignore').split('\n')


def _get_genre_raw_metadata():
    """
    Get raw lines of the movies file.
    """

    path = _get_movielens_path()

    if not os.path.isfile(path):
        _download_movielens(path)

    with zipfile.ZipFile(path) as datafile:
        return datafile.read('ml-100k/u.genre').decode(errors='ignore').split('\n')


def get_movielens_item_metadata(use_item_ids):
    """
    Build a matrix of genre features (no_items, no_features).

    If use_item_ids is True, per-item feeatures will also be used.
    """

    features = {}
    genre_set = set()
    max_genres = 0

    for line in _get_movie_raw_metadata():

        if not line:
            continue

        splt = line.split('|')
        item_id = int(splt[0])

        genres = [idx for idx, val in
                  zip(range(len(splt[5:])), splt[5:])
                  if int(val) > 0]

        if use_item_ids:
            # Add item-specific features too
            genres.append(item_id)

        for genre_id in genres:
            genre_set.add(genre_id)

        features[item_id] = genres
        if len(genres) > max_genres:
            max_genres = len(genres)
    
    mat = np.zeros((len(features) + 1, max_genres), dtype=np.uint32)
    for item_id, genre_ids in features.items():
        mat[item_id, :len(genre_ids)] = genre_ids

    return mat


def get_dense_triplets(uids, pids, nids, num_users, num_items):

    user_identity = np.identity(num_users)
    item_identity = np.identity(num_items)

    return user_identity[uids], item_identity[pids], item_identity[nids]


def get_triplets(mat):

    return mat.row, mat.col, np.random.randint(mat.shape[1], size=len(mat.row))


def get_movielens_data():
    """
    Return (train_interactions, test_interactions).
    """

    train_data, test_data = _get_raw_movielens_data()

    uids = set()
    iids = set()

    for uid, iid, rating, timestamp in itertools.chain(_parse(train_data),
                                                       _parse(test_data)):
        uids.add(uid)
        iids.add(iid)

    rows = max(uids) + 1
    cols = max(iids) + 1

    return (_build_interaction_matrix(rows, cols, _parse(train_data)),
            _build_interaction_matrix(rows, cols, _parse(test_data)))


def extract_tensorboard_metadata(log_dir):

    # Clean out and create the log dir if necessary
    shutil.rmtree(
        log_dir,
        onerror=lambda f, p, e: print("Couldn't remove %s: %s" % (p, e)))

    try:
        os.makedirs(log_dir)
    except Exception, e:
        print("Couldn't create %s: %s" % (log_dir, e))

    items_metadata = os.path.join(log_dir, 'items.txt')
    with open(items_metadata, 'w') as f:
        # There's no movie 0, but we need to add a dummy line otherwise
        # they won't line up with the embedding table
        print('0 - None', file=f)
        for line in _get_movie_raw_metadata():
            fields = line.split('|')
            if len(fields) > 1:
                print('%s - %s' % (fields[0], fields[1]), file=f)

    tags_metadata = os.path.join(log_dir, 'tags.txt')
    with open(tags_metadata, 'w') as f:
        # No need for dummy '0' as above -- this is already provided
        for line in _get_genre_raw_metadata():
            fields = line.split('|')
            if len(fields) > 1:
                # Note: fields are opposite way round from movies
                print('%s - %s' % (fields[1], fields[0]), file=f)

    return items_metadata, tags_metadata


def get_movie_names():
    output = [u'None']
    for line in _get_movie_raw_metadata():
        fields = line.split('|')
        if len(fields) > 1:
            output.append(fields[1])
    return output

