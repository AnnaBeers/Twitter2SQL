import json
import os
import networkx as nx
import pandas as pd
import pickle
import csv

from tqdm import tqdm
from datetime import date, timedelta, datetime, timezone
from psycopg2 import sql
from pprint import pprint
from collections import defaultdict
from twitter2sql.request.images import get_images_from_db, gather_images, get_top_images, \
    extract_features, cluster_features, get_image_collages

from twitter2sql.analysis.network import generate_network_gexf, stream_connection_data
from twitter2sql.core import sql_statements
from twitter2sql.request import get_timelines
from twitter2sql.core.util import clean, c, get_last_modified, \
    within_time_bounds, open_database, close_database, \
    get_column_header_dict, to_list_of_dicts, save_to_csv, \
    sql_type_dictionary, save_to_csv, to_pandas, open_tweepy_api, int_dict


def image_clustering_pipeline(cursor,
        get_tweet_ids=True,
        tweet_id_statement=None,
        tweet_ids_csv=None,
        get_image_urls=True,
        image_url_conditions=None,
        image_table='public.raw',
        image_csv=None,
        get_images=True,
        proxies=None,
        image_hash=None,
        image_directory=None,
        image_size='small',
        input_type='csv',
        extract_features=True,
        ):

    if get_tweet_ids:
        pipeline_tweet_ids(cursor, tweet_id_statement, tweet_ids_csv)

    if get_image_urls:
        pipeline_image_urls(cursor, image_url_conditions, image_table, image_csv)

    if get_images:
        gather_images(image_csv,
            image_directory,
            image_hash,
            proxies=proxies,
            size=image_size,
            input_type=input_type)

    if extract_features:
        

    return


def pipeline_tweet_ids(cursor, tweet_id_statement, tweet_ids_csv):

    print(tweet_id_statement.as_string(cursor))

    cursor.execute(tweet_id_statement)

    print('Query Finished')
    results = to_pandas(cursor)
    print(results)

    results.to_csv(tweet_ids_csv)

    return results

    
def pipeline_image_urls(cursor, image_url_conditions, image_table, image_csv):

    # TODO: Add named cursor version of this

    sql_statement = sql.SQL("""
                    SELECT x.id, x.user_id, x.created_at, x.in_reply_to_status_id,
                    jsonb_array_elements_text(x.extended_url) as extended_url,
                    jsonb_array_elements_text(x.url) as url,
                    jsonb_array_elements_text(x.extended_type) as extended_type,
                    jsonb_array_elements_text(x.type) as type
                    FROM (
                        SELECT jsonb_path_query_array(raw->'extended_entities'->'media', '$[*].media_url') AS extended_url, 
                        jsonb_path_query_array(raw->'entities'->'media', '$[*].media_url') AS url,
                        jsonb_path_query_array(raw->'extended_entities'->'media', '$[*].type') AS extended_type,
                        jsonb_path_query_array(raw->'entities'->'media', '$[*].type') AS type,
                        (raw->'user'->>'id')::bigint as user_id,
                        created_at,
                        (raw->>'in_reply_to_status_id')::bigint as in_reply_to_status_id,
                        id
                        FROM {image_table}
                        {conditions}
                    ) x
                """).format(conditions=image_url_conditions, image_table=sql.SQL(image_table))

    print(sql_statement.as_string(cursor))

    cursor.execute(sql_statement)

    print('Query Finished')
    results = to_pandas(cursor)
    results = results.drop_duplicates()
    print(results)

    results.to_csv(image_csv)

    return


if __name__ == '__main__':

    pass