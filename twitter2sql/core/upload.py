import json
import os
import re
import sys
import glob
import pytz

__UTC__ = pytz.UTC

from datetime import datetime, timezone
from psycopg2 import extras as ext
from psycopg2.extras import Json
from pprint import pprint
from tqdm import tqdm

from twitter2sql.core import sql_statements
from twitter2sql.core.util import clean, get_last_modified, \
    within_time_bounds, open_database, close_database, \
    get_column_header_dict


# CREATE_TABLE_STMT = sql_statements.CREATE_TABLE_STMT
# INSERT_TWEET_STMT = sql_statements.INSERT_TWEET_STMT


"""1. Configure parameters

Text search
This will search the full text of the tweet, any retweeted_status text, and any
 quoted_status text.

`search_text`: set to True if you want to use text search
`keywords`: add the keywords you want to match here
`all_keywords`: whether to check for all keywords. If true, it will match only
 tweets that have all keywords. If false it will check whether any of the 
 keywords exist

"""


# @profile
def upload_twitter_2_sql(database_name,
                            db_config_file,
                            twitter_json_folders,
                            table_format_csv,
                            table_name='example_table',
                            owner='example',
                            admins=[],
                            search_text=False,
                            keywords=['keyword1', 'keyword2'],
                            all_keywords=False,
                            match_dates=True,
                            start_time=None,
                            end_time=datetime.utcnow().replace(tzinfo=timezone.utc),
                            use_regex_match=False,
                            reg_expr='example_regex',
                            overwrite_db=True,
                            overwrite=True,
                            timestamp='modified',
                            json_mode='newline'):
    
    create_table_statement = sql_statements.create_table_statement(table_format_csv, table_name)
    insert_table_statement = sql_statements.insert_statement(table_format_csv, table_name)
    database, cursor = open_database(database_name, db_config_file, overwrite_db, owner, admins)

    if overwrite:
        # This errors if the tweets table does not yet exist.
        # Fix that!
        cursor.execute("DROP TABLE IF EXISTS tweets;")
        pass
    else:
        # Not sure what should happen in this case.
        print('Table already exists, and overwrite=False. Exiting.')
        return
    cursor.execute(create_table_statement)
    database.commit()

    # Add admins to the table.
    admin_add_statement = sql_statements.table_permission_statement(table_name, admins)
    cursor.execute(admin_add_statement)
    database.commit()

    column_header_dict = get_column_header_dict(table_format_csv)

    try:

        # Keep track of how many tweets have been inserted (just make sure it's running)
        total_tweets_inserted = 0

        # Process each folder
        for folder_path in twitter_json_folders:

            # Make sure only valid .json files are processed
            json_files_to_process = glob.glob(os.path.join(folder_path, '*.json'))

            # Filter to only include files within the date range:
            # this specifically filters out JSON files by the file's last modified time (UNIX timestamp), only keeping
            # files written on or after the early time bound and on or before the late time bound plus one day
            # (to allow time for tweets to be written to the file)
            if match_dates:
                
                json_files_to_process = sorted(json_files_to_process, key=lambda json_file: get_last_modified(os.path.abspath(json_file)))

                print("{} JSON files before time filtering: from {} to {}".format(
                    len(json_files_to_process), json_files_to_process[0], json_files_to_process[-1]))

                json_files_to_process = [json_file for json_file in json_files_to_process if within_time_bounds(os.path.abspath(json_file), start_time, end_time)]
                
                print("{} JSON files after time filtering: from {} to {}".format(
                    len(json_files_to_process), json_files_to_process[0], json_files_to_process[-1]))

            progress_bar = tqdm(json_files_to_process, desc='0/0 tweets inserted')
            for idx, json_file in enumerate(progress_bar):
                # For each file, extract the tweets and add the number extracted to the total_tweets_inserted
                total_tweets_inserted += extract_json_file(os.path.join(folder_path, json_file), cursor, database, keywords,
                    search_text, all_keywords, insert_table_statement, match_dates, start_time, end_time, use_regex_match, 
                    reg_expr, column_header_dict, json_mode=json_mode)

                progress_bar.set_description("{fnum}/{ftotal_tweets_inserted}: {tnum} tweets inserted".format(fnum=idx, ftotal_tweets_inserted=(len(json_files_to_process) + 1), tnum=total_tweets_inserted))
                sys.stdout.flush()

        # Close everything
        close_database(cursor, database)

    except KeyboardInterrupt:
        close_database(cursor, database)
    except Exception:
        close_database(cursor, database)
        raise

    return


# @profile
def extract_json_file(json_file_path, cursor, database, keywords, search_text, 
                            all_keywords,
                            insert_table_statement,
                            match_dates=True,
                            start_time=None,
                            end_time=datetime.utcnow().replace(tzinfo=timezone.utc),
                            use_regex_match=False,
                            reg_expr='example_regex',
                            column_header_dict=None,
                            json_mode='newline'):

    with open(json_file_path, 'r') as infile:
        queue = []

        if json_mode == 'newline':
            lines = [line for line in infile if (line and len(line) >= 2)]  # ????
        elif json_mode == 'list':
            lines = json.load(infile)

        for line in lines:

            # Load the tweet string into a dictionary.
            # There's like one tweet in one json file that is bad json, so I've just been skipping
            # it. If there end up being a lot, we should probably figure out why that's happening.
            try:
                if json_mode == 'newline':
                    tweet = json.loads(line)
                else:
                    tweet = line
                
                # Make sure that the tweet matches all filtering parameters
                if matches_parameters(tweet, search_text, keywords, all_keywords, match_dates, 
                        start_time, end_time, use_regex_match, reg_expr):
                    tweet_row = extract_tweet(tweet, column_header_dict)
                    
                    if tweet_row:
                        queue.append(tweet_row)
            
            except ValueError as e:
                print("Bad JSON")
                print("Error: {}".format(e))
                print(line)
            
        # Insert all the extracted tweets into the database
        try:
            ext.execute_batch(cursor, insert_table_statement, queue)
        except Exception as e:
            print(json_file_path)
            raise(e)
        
        # Just to keep track of how many have been inserted
        return len(queue)


# @profile
def matches_parameters(tweet, 
                        search_text=False,
                        keywords=['keyword1', 'keyword2'],
                        all_keywords=False,
                        match_dates=True,
                        start_time=None,
                        end_time=datetime.utcnow().replace(tzinfo=timezone.utc),
                        use_regex_match=False,
                        reg_expr='example_regex'):
    
    # Keyword filtering   
    if search_text:
        # Make a list of fields to check for keyword matches (could add user_description, etc.)
        keyword_texts = [get_complete_text(tweet)]

        def matches_keywords(text):
            matches = get_matching_keywords(text, keywords)

            if all_keywords:
                return matches == keywords  # only return True if all keywords matched
            else:
                return bool(matches)  # return True if there's at least one match

        keyword_matches = [matches_keywords(keyword_text) for keyword_text in keyword_texts]

        if not any(keyword_matches):
            return False

    # Time interval filtering
    
    if match_dates:
        created_at = get_nested_value(tweet, "created_at")
        created_ts = __UTC__.localize(datetime.strptime(created_at[0:19] + created_at[25:], "%a %b %d %H:%M:%S %Y"))
        
        if not created_ts or created_ts < start_time or created_ts > end_time:
            return False
    
    """
    Regex matching. This part may not be functional, review --ALB.
    """
    
    if use_regex_match:
        # Make a list of fields to check for keyword matches
        regex_texts = [get_complete_text(tweet)]
        regex_matches = [bool(re.search(reg_expr, text)) for text in regex_texts]
        if not any(regex_matches):
            return False
    
    return True


# @profile
def get_complete_text(tweet):

    """ Previously, strings in complete_text had been encoded into
        utf-8. I undid that, but there may be a reason to put that
        back in later. Used the c() function.
    """

    if 'text' in tweet:
        tweet_complete_text = tweet["text"]
    else:
        tweet_complete_text = tweet['full_text']

    if tweet["truncated"]:
        # Applicable to original tweets and commentary on quoted tweets
        tweet_complete_text = tweet["extended_tweet"]["full_text"]

    # This handles retweets of original tweets and retweets of quoted tweets
    if "retweeted_status" in tweet:
        return_text = "RT @{username}: {orig_complete_text}".format(
            username=tweet["retweeted_status"]["user"]["screen_name"],
            orig_complete_text=get_complete_text(tweet["retweeted_status"]))
        return return_text

    # I am fairly certain that the only way you can quote a tweet is by 
    # quoting the original tweet; i.e. I don't think you can quote a retweet
    elif "quoted_status" in tweet:
        return_text = "{qt_complete_text} QT @{username}: {orig_complete_text}".format(
            qt_complete_text=tweet_complete_text,
            username=tweet["quoted_status"]["user"]["screen_name"],
            orig_complete_text=get_complete_text(tweet["quoted_status"]))
        return return_text

    else:
        return tweet_complete_text


# @profile
def get_matching_keywords(search_string, keywords):

    """ This function uses regular expressions to search for keywords in a tweet.
    """

    # keyword_regex = r"(\b({reg}))|(({reg})\b)".format(reg="|".join(keywords))
    keywords = ["(\\b" + x + "\\b)" if x[0] != '#'
        else '(' + x + "\\b)" for x in keywords]
    keyword_regex = r"({reg})".format(reg="|".join(keywords))
    matches = []

    # Temporary fix for bytes/string mixing in earlier code.
    if type(search_string) is bytes:
        search_string = search_string.decode('utf-8')

    for match in re.findall(keyword_regex, search_string.lower()):
        matches = matches + list([m for m in match if m])
    matches = list(set(matches))
    return matches


# @profile
def get_nested_value_json(_dict, path, default=None):
    # Pull the nested value
    value = get_nested_value(_dict, path, default)

    # Return a string of the json dictionary
    if value:
        return json.dumps(value)


# @profile
def get_nested_value(outer_dict, path_str, default=None):

    """
    Get a value from the given dictionary by following the path
    If the path isn't valid, nothing will be returned.
    """

    # get a list of nested dictionary keys (the path)
    path = path_str.split(".")
    current_dict = outer_dict

    # step through the path and try to process it
    try:
        for step in path:
            # If it's actually a list index, convert it to an integer
            if step.isdigit():
                step = int(step)

            # Get the nested value associated with that key
            current_dict = current_dict[step]

        # Once it's at the end of the path, return the nested value
        return current_dict

    # The value didn't exist
    except (KeyError, TypeError, IndexError):
        # pprint(outer_dict)
        # print(path_str)
        # raise e
        pass

    return default


def extract_tweet(tweet, column_header_dict):
    # Adding everything to a huge tuple and inserting the tuple to the database
    # TODO: Problems here with extended entites.

    entities = tweet["entities"]
    extended_entities = None
    if 'extended_entities' in tweet:
        extended_entities = tweet['extended_entities']
    if tweet["truncated"]:
        entities = tweet["extended_tweet"]["entities"]
        if 'extended_entities' in tweet['extended_tweet']:
            extended_entities = tweet['extended_tweet']['extended_entities']
    elif "retweeted_status" in tweet:
        if tweet["retweeted_status"]["truncated"]:
            entities = tweet["retweeted_status"]["extended_tweet"]["entities"]
            if 'extended_entities' in tweet['retweeted_status']['extended_tweet']:
                extended_entities = tweet["retweeted_status"]["extended_tweet"]["extended_entities"]
        else:
            entities = tweet["retweeted_status"]["entities"]
            if 'extended_entities' in tweet['retweeted_status']:
                extended_entities = tweet["retweeted_status"]["extended_entities"]

    item = []
    for key, value in column_header_dict.items():

        if key == 'id':
            add_item = tweet["id"]
        elif key == 'complete_text':
            add_item = clean(get_complete_text(tweet))
        elif key == 'entities':
            add_item = clean(json.dumps(entities))
        elif value['type'] == 'timestamp':
                time = get_nested_value(tweet, value['instructions'])
                add_item = datetime.strptime(time[0:19] + time[25:], "%a %b %d %H:%M:%S %Y")
        else:
            if key == 'video_url_0':
                add_item = get_nested_value(entities, value["json_fieldname"])
                if add_item is not None:
                    for variant in add_item:
                        if variant['content_type'] != 'application/x-mpegURL':
                            add_item = variant['url']
            elif key == 'urls':
                add_item = get_nested_value_json(entities, value["json_fieldname"])
            elif key == 'photo':
                add_item = False
                if 'media' in entities:
                    for media in entities['media']:
                        if media['type'] == 'photo':
                            add_item = True
                if extended_entities is not None: 
                    if 'media' in extended_entities:
                        for media in extended_entities['media']:
                            if media['type'] == 'photo':
                                add_item = True
            elif key == 'video':
                add_item = False
                if 'media' in entities:
                    for media in entities['media']:
                        if media['type'] == 'video':
                            add_item = True
                if extended_entities is not None:
                    if 'media' in extended_entities:
                        for media in extended_entities['media']:
                            if media['type'] == 'video':
                                add_item = True
            elif value['instructions'] == 'entities':
                add_item = get_nested_value(entities, value["json_fieldname"])
            elif value['instructions'] == 'extended_entities':
                add_item = get_nested_value(extended_entities, value["json_fieldname"])
            elif value['instructions'] == 'dump_json':
                add_item = json.dumps(tweet)
            elif value['type'] == 'json':
                add_item = get_nested_value_json(tweet, value["json_fieldname"])
            else:
                add_item = get_nested_value(tweet, value["json_fieldname"])

            if value['clean'] and value['type'] != 'boolean':
                add_item = clean(add_item)

        item += [add_item]

    return item


if __name__ == '__main__':
    upload_twitter_2_sql()