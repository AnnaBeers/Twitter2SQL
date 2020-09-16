import tweepy
import time
import os
import json
import sys
import requests
import shutil
import csv
import numpy as np
import tables
import scipy.cluster.hierarchy as hcluster
import matplotlib.pyplot as plt
import matplotlib
import cv2

from tqdm import tqdm
from datetime import datetime
from psycopg2 import sql
from pprint import pprint
from glob import glob
from hashlib import sha256
from collections import defaultdict, Counter
from shutil import copy, rmtree
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn import preprocessing, manifold
from sklearn.decomposition import PCA
from keras.models import load_model, Model
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from umap import UMAP
from sklearn.preprocessing import StandardScaler

from twitter2sql.core import json_util
from twitter2sql.core import sql_statements
from twitter2sql.core.util import open_database, save_to_csv, \
    to_list_of_dicts, to_pandas, set_dict, int_dict, dict_dict, \
    list_dict


# @profile
def gather_images(input_data,
                output_directory,
                hash_output,
                input_type='json',
                size='large',
                hash_columns=['id', 'user_id', 'created_at'],
                original_only=False,
                proxies=None,
                timeout=None,
                overwrite=False,
                overwrite_hash=False,
                verbose=True):

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    if os.path.exists(hash_output):
        with open(hash_output, 'r') as f:
            hash_dict = json.load(f)
        hash_dict = defaultdict(dict, hash_dict)
    else:
        hash_dict = defaultdict(dict)

    hash_dict['__broken__']['tweets'] = []
    hash_dict['__broken__']['imgpath'] = None

    all_ids = []
    for key, val in hash_dict.items():
        all_ids += [x['id'] for x in val['tweets']]
    # all_ids = list(hash_dict.values())

    tweet_count = 0
    img_count = 0
    types = set()

    try:

        if input_type == 'json':
            input_jsons = glob(os.path.join(input_data, '*.json'))
            pbar = tqdm(input_jsons)

            for input_json in pbar:

                with open(input_json, 'r') as f:
                    json_data = json.load(f)

                for data in tqdm(json_data):
                    tweet_count += 1
                    pbar.set_description(f'Total tweets {tweet_count}, Total images {img_count}, Hashes {len(hash_dict)}')

                    urls = json_util.extract_images(data)
                    tweet_id = data['id_str']

                    hash_dict, img_count = save_urls(
                        urls, output_directory, hash_dict, img_count, size, proxies)

        if input_type == 'csv':
            with open(input_data, 'r') as readfile:
                row_count = sum(1 for row in readfile)

            with open(input_data, 'r') as readfile:
                tweet_ids = set()
                reader = csv.DictReader(readfile)
                pbar = tqdm(reader, total=row_count)
                for row in pbar:
                    tweet_id = row['id']
                    tweet_ids.add(tweet_id)
                    tweet_count = len(tweet_ids)
                    pbar.set_description(f'Total tweets {tweet_count}, Total images {img_count}, Hashes {len(hash_dict)}')

                    if row['extended_type'] == 'photo':
                        urls = [row['extended_url']]
                    else:
                        continue

                    hash_dict, img_count = save_urls(
                        row, urls, output_directory, hash_columns, hash_dict,
                        all_ids, img_count, size, proxies, overwrite)

    except KeyboardInterrupt as e:
        with open(hash_output, 'w') as fp:
            json.dump(hash_dict, fp)
        raise e

    with open(hash_output, 'w') as fp:
        json.dump(hash_dict, fp)

    return


def save_urls(
        data, urls, output_directory, hash_columns,
        hash_dict, all_ids, img_count, size, proxies, overwrite):

    if urls:
        for idx, url in enumerate(urls):
            tweet_id = data['id']
            img_count += 1
            ext = os.path.splitext(url)[1]
            img_code = f'{tweet_id}_{idx}'
            save_dest = os.path.join(output_directory, f'{img_code}{ext}')

            data_dict = {col: data[col] for col in hash_columns}
            data_dict['url'] = url

            if not (os.path.exists(save_dest)) or overwrite:
                # This messes up overwrite
                if img_code not in all_ids:

                    # I think this reads things twice because of the hashing, return to this.
                    r = requests.get(url + ":" + size, stream=True, proxies=proxies)
                    if r.status_code == 200:
                        r.raw.decode_content = True
                        raw_data = r.raw.data
                        sig = sha256(raw_data)
                        if sig.hexdigest() in hash_dict:
                            hash_dict[sig.hexdigest()]['tweets'] += [data_dict]
                        else:
                            hash_dict[sig.hexdigest()]['imgpath'] = os.path.abspath(save_dest)
                            hash_dict[sig.hexdigest()]['tweets'] = [data_dict]
                            with open(save_dest, "wb") as f:
                                f.write(raw_data)
                    else:
                        hash_dict['__broken__']['tweets'] += [data_dict]

    return hash_dict, img_count


def get_top_images(input_hash,
            images_directory,
            output_directory,
            top_images=50,
            remove_previous=False):

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    elif remove_previous:
        rmtree(output_directory)
        os.mkdir(output_directory)

    with open(input_hash, 'r') as f:
        hash_dict = json.load(f)  

    hash_dict = {k: v for k, v in sorted(hash_dict.items(), key=lambda item: len(set([val['user_id'] for val in item[1]])), reverse=True)}

    count = 0
    for idx, (key, item) in enumerate(hash_dict.items()):
        if key == '__broken__':
            continue
        print(key, len(item))

        # target_image = glob(os.path.join(images_directory, f'{item[0]}*'))

        for i in item:
            target_image = glob(os.path.join(images_directory, f'{i["img_code"]}*'))
            if target_image != []:
                break

        output_filename = os.path.join(output_directory, f'{str(idx).zfill(3)}_{os.path.basename(target_image[0])}')
        copy(target_image[0], output_filename)
        count += 1
        if count == top_images:
            break


def extract_features(image_directory, input_hash, output_filepath, output_hashes, limit=500):

    from keras.preprocessing import image
    from keras.applications.vgg16 import VGG16
    from keras.applications.vgg16 import preprocess_input

    with open(input_hash, 'r') as f:
        hash_dict = json.load(f)
    hash_dict.pop('__broken__', None)

    hash_vals = list(hash_dict.items())
    if limit is None:
        limit = len(hash_vals)

    base_model = VGG16(weights='imagenet', include_top=True)
    model = Model(inputs=base_model.input,
            outputs=base_model.get_layer('fc2').output)

    # Write to HDF5, large data storage format.
    hdf5_file = tables.open_file(output_filepath, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    num_cases = limit
    data_shape = (0, 4096)  # TODO: Make this adjust automatically
    data_storage = hdf5_file.create_earray(hdf5_file.root, 'features', 
        tables.Float32Atom(), shape=data_shape, filters=filters, expectedrows=num_cases)

    try:
        feature_hashes = []
        for idx, (key, item) in tqdm(list(enumerate(hash_vals))):

            if idx > limit:
                break

            target_image = item["imgpath"]
            feature_hashes += [key]

            img = image.load_img(target_image, target_size=(224, 224))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)
            output_features = model.predict(img_data)
            data_storage.append(output_features)

    except KeyboardInterrupt:
        with open(output_hashes, 'w') as fp:
            json.dump(feature_hashes, fp)
        hdf5_file.close()

    with open(output_hashes, 'w') as fp:
        json.dump(feature_hashes, fp)
    hdf5_file.close()

    return


def cluster_features(
        input_hdf5, input_hash, output_directory, image_directory,
        feature_hashes, normalize=None, verbose=True,
        remove_previous=False, show_charts=False,
        pca_components=None, n_clusters=None):

    """ This function needs a little cleaning.
    """

    if show_charts:
        matplotlib.use('TkAgg')

    with open(input_hash, 'r') as f:
        hash_dict = json.load(f)

    with open(feature_hashes, 'r') as f:
        feature_hashes = json.load(f)

    if verbose:
        print('Open data...')
        np.set_printoptions(suppress=True)

    open_hdf5 = tables.open_file(input_hdf5, "r")
    data = getattr(open_hdf5.root, 'features')
    data = data.read().reshape(data.shape[0], -1)

    if verbose and normalize:
        print('Normalize data...')
        data = preprocessing.normalize(data, norm='l2')
        data = StandardScaler().fit_transform(data)

    if verbose:
        print('Dimension Reduction...')

    if pca_components is None:

        # Take the number of PCA responsible for 90% variation.
        # Don't know if this works :)
        pca = PCA(n_components=min([100] + list(data.shape)), random_state=728).fit(data.T)
        variance = pca.explained_variance_ratio_
        sum_pca = 0
        pca_components = 0
        for var in variance:
            sum_pca += var
            pca_components += 1
            if sum_pca > .9:
                break

        if verbose:
            print(f'PCA Component Num Estimated At: {pca_components}')

        pca_features = pca.components_.T
        pca_features = pca_features[:, 0:pca_components]
    else:
        pca = PCA(n_components=min([pca_components] + list(data.shape)), random_state=728).fit(data.T)
        pca_features = pca.components_.T

    if verbose:
        print('Clustering...')

    if n_clusters is None:
        sum_of_squared_distances = []
        K = range(2, min(500, pca_features.shape[0]))

        if verbose:
            print('Estimating Cluster Num...')
            pbar = tqdm(K)
        else:
            pbar = range(K)

        for k in pbar:
            km = KMeans(n_clusters=k)
            km = km.fit(pca_features)
            sum_of_squared_distances.append(silhouette_score(pca_features, km.labels_))

        plt.plot(K, sum_of_squared_distances, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal k')

        if show_charts:
            plt.show()

        n_clusters = np.argmax(sum_of_squared_distances)

        if verbose:
            print(f'Chosen Cluster Num: {n_clusters}')
        # plt.savefig("optimal_k.png")

    clusters = KMeans(n_clusters=n_clusters, random_state=0).fit(pca_features)
    # thresh = 1
    # clusters = hcluster.fclusterdata(data, thresh, criterion="distance")
    # connectivity = kneighbors_graph(pca_features, n_neighbors=10)
    # connectivity = 0.5 * (connectivity + connectivity.T)
    # clusters = AgglomerativeClustering(n_clusters=100,
                    # linkage='ward', connectivity=connectivity).fit(pca_features)
    # clusters = DBSCAN(eps=.2).fit(pca_features)
    # clusters = SpectralClustering(n_clusters=20, eigen_solver='arpack', affinity="nearest_neighbors").fit(pca_features)

    clusters = clusters.labels_
    counts = Counter(clusters)
    counts = [[key, val] for key, val in counts.items()]
    # print(counts)
    # print(clusters)
    # print(len(set(clusters)))

    if show_charts:
        plt.figure(figsize=(45, 45))
        plt.scatter(pca_features[:, 0], pca_features[:, 1], c=clusters)
        plt.axis("equal")
        title = "threshold: %f, number of clusters: %d" % (thresh, len(set(clusters)))
        plt.title(title)
        plt.show()
    # plt.savefig("clusters.png")

    sorted_cluster_idx = np.argsort(clusters)

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    elif remove_previous:
        rmtree(output_directory)
        os.mkdir(output_directory)

    for cluster, count in tqdm(reversed(counts)):

        # print(cluster, count)
        indices = [i for i, x in enumerate(clusters) if x == cluster]

        for idx in indices:

            key = feature_hashes[idx]

            if key == '__broken__':
                continue

            cluster_directory = os.path.join(output_directory, str(cluster))
            if not os.path.exists(cluster_directory):
                os.mkdir(cluster_directory)

            target_image = hash_dict[key]["imgpath"]

            copy(target_image, os.path.join(cluster_directory, os.path.basename(target_image)))

    return


def image_scatter_plot(tsne, hash_dict, feature_hashes, image_directory, figsize=(45,45)):

    if show_charts:
        matplotlib.use('TkAgg')

    with open(input_hash, 'r') as f:
        hash_dict = json.load(f)

    with open(feature_hashes, 'r') as f:
        feature_hashes = json.load(f)

    if verbose:
        print('Open data...')
        np.set_printoptions(suppress=True)

    open_hdf5 = tables.open_file(input_hdf5, "r")
    data = getattr(open_hdf5.root, 'features')
    data = data.read().reshape(data.shape[0], -1)

    if verbose and normalize:
        print('Normalize data...')
        data = preprocessing.normalize(data, norm='l2')
        data = StandardScaler().fit_transform(data)

    # pca_features = manifold.TSNE(n_components=2, init='pca', perplexity=25,
             # random_state=0).fit_transform(data)
    # pca_features = UMAP(min_dist=.01, verbose=True, n_neighbors=20).fit_transform(data)
    # image_scatter_plot(pca_features, hash_dict, feature_hashes, image_directory)

    images = []

    for feature_hash in tqdm(feature_hashes):

        image_path = hash_dict[feature_hash]["imgpath"]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (60, 60), interpolation=cv2.INTER_AREA)
                
        images.append(image)
            
    images = np.array(images)

    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for xy, i in zip(tsne, images):
        x0, y0 = xy
        img = OffsetImage(i, zoom=1)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(tsne)
    ax.autoscale()
    plt.savefig("image_clusters.png")
    # plt.show()

    return


def get_image_collages(
        cluster_directories, output_directory='Cluster_Previews',
        image_size=(90, 90), cols=6, remove_previous=False):

    matplotlib.use('Agg')
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    elif remove_previous:
        rmtree(output_directory)
        os.mkdir(output_directory)

    cluster_directories = glob(os.path.join(cluster_directories, '*'))

    for cluster_dir in tqdm(cluster_directories):
        cluster_num = os.path.basename(cluster_dir)
        image_files = glob(os.path.join(cluster_dir, '*'))

        output_filepath = os.path.join(output_directory, f'{cluster_num}.png')

        rows = int(image_size[0] * np.ceil(len(image_files) / cols))
        collage = np.zeros((rows, cols * image_size[1], 3), dtype=float)

        row_index = 0
        col_index = 0
        for image_path in image_files:

            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)

            collage[row_index:row_index + image_size[0], col_index:col_index + image_size[1], :] = image

            if col_index == collage.shape[1] - image_size[1]:
                col_index = 0
                row_index += image_size[0]
            else:
                col_index += image_size[1]

        collage = collage.astype(int)
        plt.figure(figsize=(collage.shape[0] / 100, collage.shape[1] / 100), dpi=100, frameon=False)
        # plt.figure(figsize=(25, 25), frameon=False)
        # plt.figure(frameon=False)
        plt.margins(0, 0)
        plt.gca().set_axis_off()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.imshow(collage)

        plt.savefig(output_filepath, bbox_inches='tight', pad_inches=0.0, dpi=500)
        plt.clf()
        plt.close()

    return


def get_images_from_db(database_name,
                db_config_file,
                table_name,
                output_directory,
                conditions=None,
                limit=None,
                itersize=1000):

    raise NotImplementedError


if __name__ == '__main__':
    
    pass