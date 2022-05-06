import json
import pandas as pd

from pprint import pprint


def load_json(input_json, output_type='dict'):

    # Should probably check this at the format level.
    if input_json.endswith('json'):
        with open(input_json, 'r') as f:
            data = json.load(f)
    else:
        with open(input_json, 'r') as f:
            data = [json.loads(jline) for jline in list(f)]

    return data


def extract_mentions(input_data):

    return


def extract_images(input_data, types=['photo']):

    # Stolen from https://github.com/morinokami/twitter-image-downloader/blob/master/twt_img/twt_img.py

    if len(types) > 1:
        raise NotImplementedError

    if "media" in input_data["entities"]:

        if "extended_entities" in input_data:
            media_types = [x['type'] for x in input_data["extended_entities"]["media"]]
            extra = [
                x["media_url"] for x in input_data["extended_entities"]["media"] if x['type'] in types
            ]
        else:
            media_types = None
            extra = []

        if all([x in types for x in media_types]):
            urls = [x["media_url"] for x in input_data["entities"]["media"] if x['type'] in types]
            urls = set(urls + extra)
            return urls
    else:
        return None


if __name__ == '__main__':

    pass