import tweepy
import time
import os
import json
import sys

from tqdm import tqdm
from datetime import datetime

from twitter2sql.core.util import twitter_str_to_dt


def get_timelines(api, input_ids, output_directory, stop_condition=3200,
            append=True):

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    output_prefix = os.path.basename(output_directory)
    
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