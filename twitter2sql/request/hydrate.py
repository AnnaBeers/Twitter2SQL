import os
import json

from tqdm import tqdm


def get_user_json(api, input_ids, output_directory, ids_output_filepath=None, overwrite=True, mode='screen_name'):

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    output_prefix = os.path.basename(output_directory)

    if type(input_ids) is list:
        pass
    elif type(input_ids) is str:
        if input_ids.endswith('.txt'):
            with open(input_ids, 'r') as f:
                input_ids = f.readlines()
        else:
            raise ValueError(f"{input_ids} in str format must be a .txt file.")
    else:
        raise ValueError(f"{input_ids} must be either a filepath or a list of screen names.")

    output_file = os.path.join(output_directory, 'user_request.json')

    # Bad manual JSON formatting going on here.
    if overwrite:
        with open(output_file, 'w') as openfile:
            openfile.write('[\n')
        if ids_output_filepath is not None:
            open(ids_output_filepath, 'w').close()
    else:
        with open(sys.argv[1], "a") as openfile:
            # https://stackoverflow.com/questions/1877999/delete-final-line-in-file-with-python
            openfile.seek(0, os.SEEK_END)
            pos = openfile.tell() - 1
            while pos > 0 and file.read(1) != "\n":
                pos -= 1
                openfile.seek(pos, os.SEEK_SET)
            if pos > 0:
                openfile.seek(pos, os.SEEK_SET)
                openfile.truncate()

    total_users = 0
    for idx, uid in enumerate(tqdm(input_ids)):

        user_json = api.get_user(uid)

        if user_json:

            if ids_output_filepath is not None:
                with open(ids_output_filepath, 'a') as openfile:
                    if idx != len(input_ids) - 1:
                        openfile.write(f'{user_json._json["id"]}\n')

            with open(output_file, "a") as openfile:
                json.dump(user_json._json, openfile)

                if idx == len(input_ids) - 1:
                    openfile.write('\n')
                else:
                    openfile.write(",\n")

            total_users += 1

    with open(output_file, "a") as openfile:
        openfile.write("]")

    print(f'{total_users} users collected out of {len(input_ids)}')

    return


def get_historic_tweets_before_id(api, uid, max_id, stop_condition):

    # List of tweets we've collected so far
    tweets = []
    finished = False

    # The timeline is returned as pages of tweets (each page has 20 tweets, starting with the 20 most recent)
    # If a cap has been set and our list of tweets gets to be longer than the cap, we'll stop collecting
    cursor_args = {"id": uid, "count": 200}
    if max_id:
        cursor_args["max_id"] = max_id

    page_num = 0

    try:
        for page in tweepy.Cursor(api.user_timeline, tweet_mode='extended', 
                    **cursor_args).pages(16):
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

            print(page_num)
            page_num += 1

            if finished:
                break

    except tweepy.error.TweepError as ex:
        # We received a rate limiting error, so wait 15 minutes
        if "429" in str(ex):  # a hacky way to see if it's a rate limiting error
            time.sleep(15 * 60)
            print("rate limited :/")

            # Try again
            return self.get_historic_tweets_before_id(api, uid, max_id)

        elif any(code in str(ex) for code in ["401", "404"]):
            return (None, True, [])

        else:
            print(uid)
            print(ex)
            return (None, True, [])

    if tweets:
        max_id = max(tweets, key=lambda t: int(t["id_str"]))
        return (max_id, finished, tweets)

    else:
        return (None, True, [])


def extract_users_from_tweet_jsons():

    return


if __name__ == '__main__':

    pass