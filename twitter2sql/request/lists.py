import tweepy
import csv
import os

from pprint import pprint
from tqdm import tqdm


def get_list_members(api, user, list_id, output_csv_folder):
    members = []

    if not os.path.exists(output_csv_folder):
        os.mkdir(output_csv_folder)

    descriptors = ['name', 'description', 'slug', 'created_at', 'id']
    for list_object in api.lists_all(user):
        slug = list_object._json['slug']
        print(slug, list_object._json['id'])
        try:
            users = []
            for result in tweepy.Cursor(api.list_members, user, slug).pages():
                for page in result:
                    users += [[page._json['id'], page._json['screen_name'], page._json['name']]]
            with open(os.path.join(output_csv_folder, f'{slug}.csv'), 'w') as writefile:
                writer = csv.writer(writefile, delimiter=',')
                for row in users:
                    print(row)
                    writer.writerow(row)
        except Exception as e:
            print(f'Failed on {slug}')
            print(e)

    return None


def add_to_list(api, input_ids, list_id):
    
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

    for idx, uid in enumerate(tqdm(input_ids)):
        print(list_id, uid)
        try:
            api.add_list_member(list_id=str(list_id), user_id=uid)
        except Exception as e:
            print(e)

    return None


if __name__ == '__main__':
    pass

