import tweepy
import time
import os
import json
import sys
import requests
import shutil

from tqdm import tqdm
from datetime import datetime
from psycopg2 import sql
from pprint import pprint
from glob import glob
from hashlib import sha256
from collections import defaultdict
from shutil import copy

from twitter2sql.core import json_util
from twitter2sql.core import sql_statements
from twitter2sql.core.util import open_database, save_to_csv, \
        to_list_of_dicts, to_pandas, set_dict, int_dict, dict_dict


# @profile
def get_images_from_json(input_directory,
                output_directory,
                hash_output,
                size='large',
                proxies=None,
                timeout=None,
                overwrite=False,
                overwrite_hash=False,
                verbose=True):

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    input_jsons = glob(os.path.join(input_directory, '*.json'))
    
    if os.path.exists(hash_output):
        with open(hash_output, 'r') as f:
            # hash_dict = json.loads(f)
            hash_dict = json.load(f)
        hash_dict = defaultdict(list, hash_dict)
    else:
        hash_dict = defaultdict(list)

    all_ids = []
    for key, val in hash_dict.items():
        all_ids += val

    tweet_count = 0
    img_count = 0
    pbar = tqdm(input_jsons)
    types = set()

    try:
        for input_json in pbar:

            with open(input_json, 'r') as f:
                json_data = json.load(f)

            for data in tqdm(json_data):
                tweet_count += 1
                pbar.set_description(f'Total tweets {tweet_count}, Total images {img_count}, Hashes {len(hash_dict)}')
                if not data['in_reply_to_status_id']:
                    continue
                urls = json_util.extract_images(data)
                tweet_id = data['id_str']

                if urls:
                    for idx, url in enumerate(urls):
                        img_count += 1
                        ext = os.path.splitext(url)[1]
                        img_code = f'{tweet_id}_{idx}'
                        save_dest = os.path.join(output_directory, f'{img_code}{ext}')

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
                                        hash_dict[sig.hexdigest()] += [img_code]
                                    else:
                                        hash_dict[sig.hexdigest()] += [img_code]
                                        with open(save_dest, "wb") as f:
                                            f.write(raw_data)
                                else:
                                    hash_dict['__broken__'] += [img_code]

    except KeyboardInterrupt as e:
        with open(hash_output, 'w') as fp:
            json.dump(hash_dict, fp)
        raise e 

    with open(hash_output, 'w') as fp:
        json.dump(hash_dict, fp)

    return


def get_top_images(input_hash, 
            images_directory,
            output_directory,
            top_images=50):

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    with open(input_hash, 'r') as f:
        hash_dict = json.load(f)  

    hash_dict = {k: v for k, v in sorted(hash_dict.items(), key=lambda item: len(item[1]), reverse=True)}

    count = 0
    for key, item in hash_dict.items():
        if key == '__broken__':
            continue
        print(key, len(item))
        target_image = glob(os.path.join(images_directory, f'{item[0]}*'))
        print(target_image)
        copy(target_image[0], os.path.join(output_directory, os.path.basename(target_image[0])))
        count += 1
        if count == top_images:
            break


def get_images_from_db(database_name,
                db_config_file,
                table_name,
                output_directory,
                conditions=None,
                limit=None,
                itersize=1000):

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    database, cursor = open_database(database_name, db_config_file, 
            named_cursor='image_retrieval', itersize=itersize)

    output_prefix = os.path.basename(output_directory)
    
    # cursor.execute(sql_statements.count_rows(table_name))
    # print(cursor.fetchall())
    # return
    conditions = sql_statements.format_conditions(conditions)
    limit_statement = sql_statements.limit(limit)

    image_statement = sql.SQL("""
        SELECT {select}
        FROM {table_name}
        {conditions}
        {limit_statement}
        """).format(table_name=sql.SQL(table_name),
                select=sql.SQL('*'), conditions=conditions,
                limit_statement=limit_statement)
    print(image_statement.as_string(cursor))
    cursor.execute(image_statement)
    
    count = 0
    progress_bar = tqdm()
    while True:
        result = cursor.fetchmany(cursor.itersize)
        if result:
            for item in result:
                item = dict(item)
                pprint(item)
        else:
            cursor.close()
            break

    return

    if type(input_ids) is list:
        pass
    elif type(input_ids) is str:
        if input_ids.endswith('.txt'):
            with open(input_ids, 'r') as f:
                input_ids = [line.rstrip() for line in f]
        else:
            raise ValueError(f"{input_ids} in str format must be a .txt file.")
    else:
        raise ValueError(f"{input_ids} must be either a filepath or a list of screen names.")

    total_tweets = 0
    with tqdm(input_ids) as t:
        for uid in t:

            output_file = os.path.join(output_directory, f'{output_prefix}_{str(uid)}.json')

            if stop_condition == 'last_tweet':
                if not os.path.exists(output_file):
                    print(output_file)
                    # If it's the first time, just grab the maximum tweets.
                    tweets = get_historic_tweets(api, uid, 3200, t)
                elif append:
                    print(output_file)
                    # Otherwise read the last tweet and gather from there.
                    # May error if someone modifies the json.
                    with open(output_file, 'r') as f:
                        tweets = json.load(f)
                        last_tweet_date = twitter_str_to_dt(tweets[-1]['created_at'])
                        tweets = get_historic_tweets(api, uid, last_tweet_date, t)
                else:
                    tweets = None
            else:
                tweets = get_historic_tweets(api, uid, stop_condition, t)

            if tweets:
                
                if os.path.exists(output_file):
                    with open(output_file, "r+") as openfile:
                        # https://stackoverflow.com/questions/1877999/delete-final-line-in-file-with-python
                        openfile.seek(0, os.SEEK_END)
                        pos = openfile.tell() - 1
                        while pos > 0 and openfile.read(1) != "\n":
                            pos -= 1
                            openfile.seek(pos, os.SEEK_SET)
                        if pos > 0:
                            openfile.seek(pos, os.SEEK_SET)
                            openfile.truncate()
                            openfile.write(',\n')
                    with open(output_file, "a") as openfile:
                        for idx, tweet in enumerate(tweets):
                            json.dump(tweet, openfile)
                            if idx == len(tweets) - 1:
                                openfile.write('\n')
                            else:
                                openfile.write(",\n")
                        openfile.write("]")
                else:
                    # with open(output_file, "w") as f:
                        # json.dump(tweets, f) 
                    with open(output_file, "a") as openfile:
                        openfile.write("[\n")
                        for idx, tweet in enumerate(tweets):
                            json.dump(tweet, openfile)
                            if idx == len(tweets) - 1:
                                openfile.write('\n')
                            else:
                                openfile.write(",\n")
                        openfile.write("]")

                total_tweets += len(tweets)

    print(f'{total_tweets} tweets collected.')

    return tweets


# Get a uid's tweets
def get_historic_tweets(api, uid, stop_condition, progress_bar):
    max_id, finished, tweets = None, False, []
    while not finished:
        max_id, finished, returned_tweets = get_historic_tweets_before_id(api, uid, max_id, stop_condition, progress_bar)

        if returned_tweets:
            tweets.extend(returned_tweets)

    return tweets


# Gets 3200 of the most recent tweets associated with the given uid before before_id (or the 3200 most recent tweets if before_id is None)
# Returns the minimum id of the list of tweets (i.e. the id corresponding to the earliest tweet)
def get_historic_tweets_before_id(api, uid, max_id, stop_condition, progress_bar):

    # The timeline is returned as pages of tweets (each page has 20 tweets, starting with the 20 most recent)
    # If a cap has been set and our list of tweets gets to be longer than the cap, we'll stop collecting
    iterator_count = 200
    cursor_args = {"id": uid, "count": iterator_count}
    if max_id:
        cursor_args["max_id"] = max_id

    try:
        tweets, finished = collect_timeline(api, cursor_args, iterator_count,
            stop_condition, progress_bar)

    except tweepy.error.TweepError as ex:
        # We received a rate limiting error, so wait 15 minutes
        if "429" in str(ex):  # a hacky way to see if it's a rate limiting error
            time.sleep(15 * 60)
            print("rate limited :/")

            # Try again
            tweets, finished = collect_timeline(api, cursor_args, 
                    iterator_count, stop_condition, progress_bar)

        elif any(code in str(ex) for code in ["401", "404"]):
            print(ex)
            return (None, True, [])

        else:
            print(uid)
            print(ex)
            return (None, True, [])

    if tweets:
        tweets.sort(reverse=False, key=lambda t: twitter_str_to_dt(t['created_at']))
        max_id = max(tweets, key=lambda t: int(t["id_str"]))
        return (max_id, finished, tweets)

    else:
        return (None, True, [])


def collect_timeline(api, cursor_args, iterator_count, stop_condition,
            progress_bar):

    # List of tweets we've collected so far
    tweets = []
    finished = False
    page_num = 0

    for page in tweepy.Cursor(api.user_timeline, tweet_mode='extended', 
                **cursor_args).pages(3200 / iterator_count):
        # Adding the tweets to the list

        json_tweets = [tweet._json for tweet in page]

        finished, filtered_tweets = check_if_collection_is_finished(json_tweets, stop_condition)

        if finished:
            # Filter out any older tweets
            json_tweets = filtered_tweets
        else:
            # We get 900 requests per 15-minute window, or 1 request/second, so wait 1 second between each request just to be safe
            time.sleep(1)

        tweets.extend(json_tweets)

        progress_bar.set_description(f'Tweets: {iterator_count * (page_num + 1)}')
        page_num += 1

        if finished:
            break

    return tweets, finished


def check_if_collection_is_finished(tweets, stop_condition):
    finished, filtered_tweets = False, []

    if isinstance(stop_condition, datetime):
        min_tweet = min(tweets, key=lambda t: twitter_str_to_dt(t["created_at"]))
        min_date = twitter_str_to_dt(min_tweet["created_at"])
        if min_date < stop_condition:
            finished, filtered_tweets = True, [t for t in tweets if twitter_str_to_dt(t["created_at"]) >= stop_condition]
        elif len(tweets) >= 3200:
            tweets.sort(reverse=True, key=lambda t: twitter_str_to_dt(t['created_at']))
            finished, filtered_tweets = True, tweets[:stop_condition]
    elif len(tweets) >= stop_condition:
        tweets.sort(reverse=True, key=lambda t: twitter_str_to_dt(t['created_at']))
        finished, filtered_tweets = True, tweets[:stop_condition]

    # elif self.timebound_type == "date":
    #     min_tweet = min(tweets, key=lambda t: utils.twitter_str_to_dt(t["created_at"]))
    #     min_date = utils.twitter_str_to_dt(min_tweet["created_at"])
    #     if min_date < self.timebound_arg:
    #         finished, filtered_tweets = True, [t for t in tweets if utils.twitter_str_to_dt(t["created_at"]) >= self.timebound_arg]

    # elif self.timebound_type == "last_tweet":
    #     min_tweet_id = int(min(tweets, key=lambda t: int(t["id_str"]))["id_str"])
    #     if min_tweet_id <= self.most_recent_tweet_id:
    #         finished, filtered_tweets = True, [t for t in tweets if int(t["id_str"]) > self.most_recent_tweet_id]
    # else:
    #     raise ValueError("{} isn't a supported parameter.".format(self.timebound_type))

    return finished, filtered_tweets


if __name__ == '__main__':
    
    pass