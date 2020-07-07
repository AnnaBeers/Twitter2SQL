import pickle
import os
import pandas as pd
import lxml.etree as etree
import csv
import networkx as nx

from psycopg2 import sql
from pprint import pprint
from collections import defaultdict
from tqdm import tqdm
from datetime import date, timedelta, datetime, timezone

from twitter2sql.core import sql_statements
from twitter2sql.core.util import clean, c, get_last_modified, \
    within_time_bounds, open_database, close_database, \
    get_column_header_dict, to_list_of_dicts, save_to_csv, \
    sql_type_dictionary


def export_reply_threads(database_name, 
        db_config_file, 
        table_name,
        select_columns=['tweet', 'user_id', 'user_name',
         'user_screen_name', 'created_at', 
        'in_reply_to_user_id', 'in_reply_to_user_screen_name',
        'in_reply_to_status_id'],
        seed_conditions=None,
        seed_db_config_file=None,
        seed_database_name=None,
        seed_table_name=None,
        seed_limit=500,
        seed_random_percent=None,
        reply_range=[1, 5],
        verbose=False,
        output_type='csv',
        output_filepath=None):

    if seed_database_name is None:
        seed_database_name = database_name
    if seed_db_config_file is None:
        seed_db_config_file = db_config_file

    seed_database, seed_cursor = open_database(seed_database_name, seed_db_config_file)
    database, cursor = open_database(database_name, db_config_file)

    if 'id' not in select_columns:
        select_columns = select_columns + ['id']

    seed_posts = get_seed_posts(seed_cursor, seed_limit, seed_conditions, select_columns,
            seed_random_percent, seed_table_name)

    if output_type == 'csv':
        with open(output_filepath, 'w') as openfile:
            writer = csv.writer(openfile, delimiter=',')
            writer.writerow(select_columns + ['path', 'depth', 'is_seed'])
     
            for seed_post in tqdm(seed_posts):

                results = get_reply_thread(cursor, seed_post, table_name, select_columns,
                        reply_range)

                writer.writerow([seed_post[key] for key in select_columns] + ['NONE', 1, 'TRUE'])

                if results:
                    for idx, result in enumerate(results):
                        if result['depth'] == 1 or result['depth'] > 2:
                            continue
                        else:
                            construct_tree(result, results, idx, writer, select_columns)

    elif output_type == 'networkx':
        with open(output_filepath, 'wb') as openfile:

            thread_networks = []

            for seed_post in tqdm(seed_posts):

                results = get_reply_thread(cursor, seed_post, table_name, select_columns,
                        reply_range)

                G = nx.Graph(id=seed_post['id'])
                G.add_node(seed_post['id'], time=seed_post['created_at'], user_id=seed_post['user_id'], depth=1)

                if results:
                    for idx, result in enumerate(results):
                        G.add_node(result['id'], time=result['created_at'], user_id=result['user_id'], depth=result['depth'])
                        # print(G.nodes(data=True))
                        G.add_edge(result['id'], result['in_reply_to_status_id'])
                        # print(G.nodes(data=True))

                thread_networks += [G]

            pickle.dump(thread_networks, openfile)


def get_seed_posts(cursor, seed_limit, seed_conditions, 
            select_columns, seed_random_percent, seed_table_name):

    seed_limit = sql_statements.limit(seed_limit) 
    select_columns_sql = sql_statements.select_cols(select_columns)

    sql_statement = sql.SQL("""
        SELECT {select_columns}
        FROM {table_name}
        {random}
        {seed_conditions}
        {seed_limit}
        """).format(table_name=sql.SQL(seed_table_name),
                random=sql_statements.random_sample(seed_random_percent),
                seed_limit=seed_limit,
                seed_conditions=seed_conditions,
                select_columns=select_columns_sql)
    print(sql_statement.as_string(cursor))

    cursor.execute(sql_statement)  
    seed_posts = to_list_of_dicts(cursor)

    return seed_posts


def get_reply_thread(cursor, seed_post, table_name, select_columns,
            reply_range):

    post_id = seed_post['id']
    # print(seed_post)

    for col in ['id', 'in_reply_to_status_id']:
        if col not in select_columns:
            select_columns += [col]
    select_columns.insert(0, select_columns.pop(select_columns.index('id')))

    select_columns_sql = sql_statements.select_cols(select_columns)
    select_columns_child = sql_statements.select_cols(['c.' + col for col in select_columns])

    types = sql_type_dictionary()
    column_types = ''
    for col in select_columns:
        if col == 'id':
            continue
        else:
            column_types += 'NULL::{} AS {},'.format(types[col], col)
    column_types = sql.SQL(column_types)

    sql_statement = sql.SQL("""
        WITH RECURSIVE recursive_tweets({select_columns}, depth, path) AS (
        SELECT {seed_post}::bigint AS id, 
        {column_types}
        1::INT AS depth,
        {seed_post}::TEXT AS path
        UNION ALL
        SELECT {select_columns_child}, p.depth + 1 AS depth, (p.path || '->' || c.id::TEXT) 
        FROM recursive_tweets AS p, {table_name} AS c WHERE c.in_reply_to_status_id = p.id
        )
        SELECT * FROM recursive_tweets AS n;
        """).format(table_name=sql.SQL(table_name), seed_post=sql.SQL(str(post_id)),
                select_columns=select_columns_sql, select_columns_child=select_columns_child,
                column_types=column_types)

    cursor.execute(sql_statement)
    results = to_list_of_dicts(cursor)
    results.pop(0)

    results_length = len(results)
    if reply_range is not None:
        if reply_range[0] <= results_length <= reply_range[1]:
            pass
        else:
            return None

    return results


def construct_tree(row, results, idx, writer, select_columns):
    depth = row['depth']
    parent_id = row['in_reply_to_status_id']
    writer.writerow([row[key] for key in select_columns + ['path', 'depth']] + ['FALSE'])
    for sub_idx, result in enumerate(results[idx:]):
        if result['depth'] == depth and result['in_reply_to_status_id'] == parent_id:
            continue
        elif result['in_reply_to_status_id'] == row['id']:
            construct_tree(result, results, idx + sub_idx, writer, select_columns)

    return


"""
    old_statment = sql.SQL(EXPLAIN
                WITH RECURSIVE tweets_cte(id, user_id, tweet, user_name, user_screen_name,
                parent_id,
                created_at, in_reply_to_user_screen_name,
                in_reply_to_user_id, depth, path) AS (
                SELECT {seed_post}::bigint AS id, NULL::bigint AS user_id,
                NULL::TEXT as tweet, 
                NULL::TEXT as user_name, NULL::TEXT as user_screen_name,
                NULL::bigint AS in_reply_to_status_id,
                NULL::timestamptz AS created_at, NULL::text as in_reply_to_user_screen_name,
                NULL::bigint as in_reply_to_user_id, 1::INT AS depth,
                {seed_post}::TEXT AS path
                UNION ALL
                SELECT c.id, c.user_id, c.tweet, c.user_name, c.user_screen_name, c.in_reply_to_status_id, c.created_at, c.in_reply_to_user_screen_name,
                c.in_reply_to_user_id, p.depth + 1 AS depth, (p.path || '->' || c.id::TEXT) 
                FROM tweets_cte AS p, {table_name} AS c WHERE c.in_reply_to_status_id = p.id
                )
                SELECT * FROM tweets_cte AS n;
                ).format(table_name=sql.SQL(table_name), seed_post=sql.SQL(str(post_id)),
                select_columns=select_columns_sql, select_columns_child=select_columns_child,
                column_types=column_types)
                """


if __name__ == '__main__':

    pass