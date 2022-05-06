# coding: utf-8


import requests
import csv
import json
import multiprocessing as mp
import os
import psycopg2
import re
import requests
import sys
import time

from psycopg2 import extras as ext
from datetime import datetime
from tqdm import tqdm
from functools import partial
from pprint import pprint
from urllib.parse import urlparse

from twitter2sql.core import sql_statements
from twitter2sql.core.util import clean, c, get_last_modified, \
    within_time_bounds, open_database, close_database, \
    get_column_header_dict


def unwind_urls(database_name,
                db_config_file,
                twitter_json_folders,
                input_table_name,
                output_table_name,
                admins,
                cache_dir='./url_unwind_cache',
                column_headers={'expanded_url_0': ['TEXT', ''], 
                'resolved_url': ['TEXT', ''], 'domain': ['TEXT', '']},
                chunk_size=1000,
                overwrite_urls=False,
                overwrite_table=False):

    """ chunk_size: how many URLs per chunk - can be adjusted
    """

    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    # Open database
    database, cursor = open_database(database_name, db_config_file)

    # Create table
    create_table_statement = sql_statements.create_table_statement(
        column_headers, output_table_name)

    if overwrite_table:
        cursor.execute(sql_statements.drop_table_statement(output_table_name))

    cursor.execute(create_table_statement)
    database.commit()

    # Add admins to the table.
    admin_add_statement = sql_statements.table_permission_statement(
        output_table_name, 
        admins)
    cursor.execute(admin_add_statement)
    database.commit()

    # Collect existing URLs from table.
    if not os.path.isfile(os.path.join(cache_dir, 'distinct_urls_' + 
                database_name + ".json")) or overwrite_urls:
        
        not_null_statement = sql_statements.not_null_statement(
            input_table_name, 'expanded_url_0', select="expanded_url_0", 
            distinct='expanded_url_0')
        cursor.execute(not_null_statement)
        short_urls = sorted([u[0] for u in cursor.fetchall()])

        with open(os.path.join(cache_dir, "distinct_urls_" + database_name + 
                    ".json"), "w+") as f:
            json.dump(short_urls, f)

    else:
        with open(os.path.join(cache_dir, "distinct_urls_" + database_name +
                 ".json")) as f:
            short_urls = json.load(f)

    # Split URLs into chunks
    url_chunks = []
    for index in range(0, len(short_urls), chunk_size):
        chunk_id = index // chunk_size
        url_chunks += [(chunk_id, short_urls[index: (index + chunk_size)])]
    total_chunks = len(url_chunks)

    # Feed those chunks to parallel processes.
    func = partial(process_chunk, cache_dir, total_chunks, overwrite_urls)
    pool = mp.Pool()
    _ = pool.map_async(func, url_chunks)
    pool.close()
    pool.join()

    # Make sure we've got them all
    for chunk_id, _ in url_chunks:
        assert os.path.isfile(os.path.join(cache_dir, f"{chunk_id}.json")), \
            "No file exists for chunk {}".format(chunk_id)

    # Compile all URL chunks into one big file
    compiled_urls = []
    for chunk_id, _ in url_chunks:
        fname = os.path.join(cache_dir, f"{chunk_id}.json")
        with open(fname) as f:
            chunk_urls = json.load(f)
            compiled_urls.extend(chunk_urls)

    insert_table_statement = sql_statements.insert_statement(column_headers,
        output_table_name)

    ext.execute_batch(cursor, insert_table_statement, compiled_urls)
    close_database(cursor, database)

    return


def process_chunk(cache_dir, total_chunks, overwrite, chunk):

    chunk_id, urls_to_expand = chunk

    # Only expand the URLs if we haven't already expanded and cached them
    if not os.path.isfile(os.path.join(cache_dir, f"{chunk_id}.json")) or overwrite:
        print("{}/{}".format(chunk_id, total_chunks))
        expanded_urls = []
        for short_url in tqdm(urls_to_expand):
            expanded_urls += [expand_url(short_url, short_url, url_timeout=60)]

        # Cache the chunk so if something goes wrong later we don't have to expand everything again
        with open(os.path.join(cache_dir, "{}.json".format(chunk_id)), "w+") as f:
            json.dump(expanded_urls, f)


def get_domain(url):

    """ Returns the domain, or None if the URL is invalid
    """

    p = urlparse(url)
    if p.netloc:
        return p.netloc.lower()


def expand_url(orig_url, short_url, url_timeout=60, domain_equivalence=False):

    ## Resolves the URL, but returns None if the expanded_url is invalid or couldn't be resolved.
    ## The `url_timeout` parameter can be used to specify how many seconds to wait for each URL.
    ## If domain_equivalence = True (default False), we'll stop as soon as the URL domain remains constant.
    ## Otherwise, keep unwinding until the entire URL is equivalent
    '''
    Logic:
    (1) expanded_url = follow_url(short_url)
    (2) if expanded_url is not a valid URL, return short_url
    (3) if expanded_url is a valid URL and but isn't a redirect or expanded_url == short_url
            we assume short_url has already been completely expanded and return short_url
    (4) otherwise set short_url <- expanded_url and repeat from beginning
    Returns: (original_url, expanded_url (or None), expanded_domain (or None))
    '''
        
    # (0) if it's a twitter.com domain, the URL has already been fully expanded
    # so return it
    short_domain = get_domain(short_url)
    if short_domain == "twitter.com":
        return (short_url, short_url, short_domain)

    # (1) expanded_url = follow_url(short_url)
    expanded_url = None
    try:
        response = requests.head(short_url, timeout=url_timeout)
        if response.is_redirect:
            expanded_url = response.headers["location"]
        else:
            expanded_url = short_url

    except requests.exceptions.ConnectionError:
        print("Connection error: {}".format(short_url))
        return (orig_url, None, None)

    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        return (orig_url, None, None)
    
    # We weren't able to expand the URL, so the URL is broken
    # When would this happen?? --ALB
    if expanded_url is None:
        return (orig_url, None, None)

    expanded_domain = get_domain(expanded_url)

    # The expanded_url isn't valid
    if expanded_domain is None:
        return (orig_url, None, None)

    # Expanding the URL took us to the same place, so it was already completely
    # expanded
    if domain_equivalence:
        if short_domain == expanded_domain:
            return (orig_url, short_url, short_domain)
    else:
        if short_url == expanded_url:
            return (orig_url, short_url, short_domain)
    
    # Otherwise, following the URL took us to a new URL or domain so start 
    # again with the new URL to see if there's more expansion to do
    return expand_url(orig_url, expanded_url, url_timeout=url_timeout, 
        domain_equivalence=domain_equivalence)


if __name__ == '__main__':
    pass