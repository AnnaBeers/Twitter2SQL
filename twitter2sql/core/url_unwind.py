# coding: utf-8

from urllib.parse import urlparse
import requests

import csv
from datetime import datetime
import json
import multiprocessing as mp
import os
import psycopg2
from psycopg2 import extras as ext
import re
import requests
import sys
import time

from tqdm import tqdm
from functools import partial

from twitter2sql.core import sql_statements
from twitter2sql.core.util import clean, c, get_last_modified, within_time_bounds, \
    open_database, close_database, get_column_header_dict

#############################
## URL expansion functions ##
#############################

# ## Returns the domain, or None if the URL is invalid
# def get_domain(url):
#     p = urlparse(url)
#     if p.netloc:
#         return p.netloc.lower()

# ## Resolves the URL, but returns None if the expanded_url is invalid or couldn't be resolved.
# ## The `url_timeout` parameter can be used to specify how many seconds to wait for each URL.
# ## If domain_equivalence = True (default False), we'll stop as soon as the URL domain remains constant.
# ## Otherwise, keep unwinding until the entire URL is equivalent
# '''
# Logic:
# (0) if it's a twitter.com domain, the URL has already been fully expanded, so return it
# (1) expanded_url = follow_url(short_url)
# (2) if expanded_url is not a valid URL, return short_url
# (3) if expanded_url is a valid URL and but isn't a redirect or expanded_url == short_url
#         we assume short_url has already been completely expanded and return short_url
# (4) otherwise set short_url <- expanded_url and repeat from beginning
# Returns: (original_url, expanded_url (or None), expanded_domain (or None))
# '''
# def expand_url(orig_url, short_url, url_timeout=60, domain_equivalence=False):
    
#     short_domain = get_domain(short_url)
#     if short_domain == "twitter.com":
#         return (short_url, short_url, short_domain)

#     expanded_url = None
#     try:
#         response = requests.head(short_url, timeout=url_timeout)
#         if response.is_redirect:
#             expanded_url = response.headers["location"]
#         else:
#             expanded_url = short_url

#     except requests.exceptions.ConnectionError:
#         print("connection error: {}".format(short_url))
#         return (orig_url, None, None)

#     except Exception as ex:
#         template = "An exception of type {0} occurred. Arguments:\n{1!r}"
#         message = template.format(type(ex).__name__, ex.args)
#         print(message)
#         return (orig_url, None, None)
    
#     ## We weren't able to expand the URL, so the URL is broken
#     if not expanded_url:
#         return (orig_url, None, None)

#     expanded_domain = get_domain(expanded_url)

#     ## The expanded_url isn't valid
#     if not expanded_domain:
#         return (orig_url, None, None)

#     ## Expanding the URL took us to the same place, so it was already completely expanded
#     if domain_equivalence:
#         if short_domain == expanded_domain:
#             return (orig_url, short_url, short_domain)
#     else:
#         if short_url == expanded_url:
#             return (orig_url, short_url, short_domain)
    
#     ## Otherwise, following the URL took us to a new URL or domain so start again with the new URL to see if there's more expansion to do
#     return expand_url(orig_url, expanded_url, url_timeout=url_timeout, domain_equivalence=domain_equivalence)


# ########################
# ## Database functions ##
# ########################

# ###################################################################################
# ## 3. Start parallel jobs to process each chunk with as many cores as we can get ##
# ###################################################################################


# ###############################
# ## 4. Compile all URL chunks ##
# ###############################

# # Make sure we've got them all
# for chunk_id, _ in url_chunks:
#     assert os.path.isfile("cache_1/{}.json".format(chunk_id)), "No file exists for chunk {}".format(chunk_id)

# ## Compile all URL chunks into one big file
# compiled_urls = []
# for chunk_id, _ in url_chunks:
#     fname = "cache_1/{}.json".format(chunk_id)
#     with open(fname) as f:
#         chunk_urls = json.load(f)
#         compiled_urls.extend(chunk_urls)


# ###############################
# ## 5. Load into the database ##
# ###############################

# ext.execute_batch(cursor, INSERT_TWEET_STMT, compiled_urls)


def unwind_urls(database_name,
                db_config_file,
                twitter_json_folders,
                input_table_name,
                output_table_name,
                cache_dir='./url_unwind_cache',
                column_headers={'expanded_url_0': ['TEXT', ''], 'resolved_url': ['TEXT', ''], 'domain': ['TEXT', '']},
                chunk_size=1000,
                overwrite=False):

    """ chunk_size: how many URLs per chunk - can be adjusted
    """

    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    create_table_statement = sql_statements.create_table_statement(column_headers, output_table_name)
    insert_table_statement = sql_statements.insert_statement(column_headers, output_table_name)
    database, cursor = open_database(database_name, db_config_file)

    cursor.execute(create_table_statement)
    database.commit()

    if not os.path.isfile(os.path.join(cache_dir, 'distinct_urls_' + database_name + ".json")) or overwrite:
        
        not_null_statement = sql_statements.not_null_statement(input_table_name, 'expanded_url_0', select="expanded_url_0", distinct='expanded_url_0')
        cursor.execute(not_null_statement)
        short_urls = sorted([u[0] for u in cursor.fetchall()])

        with open(os.path.join(cache_dir, "distinct_urls_" + database_name + ".json"), "w+") as f:
            json.dump(short_urls, f)

    else:
        with open(os.path.join(cache_dir, "distinct_urls_" + database_name + ".json")) as f:
            short_urls = json.load(f)

    url_chunks = []
    for index in range(0, len(short_urls), chunk_size):
        chunk_id = index // chunk_size
        url_chunks += [(chunk_id, short_urls[index: (index + chunk_size)])]
    total_chunks = len(url_chunks)

    func = partial(process_chunk, cache_dir, total_chunks, overwrite)
    pool = mp.Pool()
    _ = pool.map_async(func, url_chunks)
    pool.close()
    pool.join()

    close_database(cursor, database)

    return


def process_chunk(cache_dir, total_chunks, overwrite, chunk):

    chunk_id, urls_to_expand = chunk
    print("{}/{}".format(chunk_id, total_chunks))

    # Only expand the URLs if we haven't already expanded and cached them
    if not os.path.isfile(os.path.join(cache_dir, "{}.json".format(chunk_id))) or overwrite:

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


def expand_url(orig_url, short_url, url_timeout=60, domain_equivalence=False):
    
    # (0) if it's a twitter.com domain, the URL has already been fully expanded, so return it
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

    # Expanding the URL took us to the same place, so it was already completely expanded
    if domain_equivalence:
        if short_domain == expanded_domain:
            return (orig_url, short_url, short_domain)
    else:
        if short_url == expanded_url:
            return (orig_url, short_url, short_domain)
    
    # Otherwise, following the URL took us to a new URL or domain so start again with the new URL to see if there's more expansion to do
    return expand_url(orig_url, expanded_url, url_timeout=url_timeout, domain_equivalence=domain_equivalence)


if __name__ == '__main__':
    pass