import psycopg2
import random
import csv
import datetime

from psycopg2 import sql
from pprint import pprint
from collections import defaultdict

from twitter2sql.core import sql_statements
from twitter2sql.core.util import clean, c, get_last_modified, \
    within_time_bounds, open_database, close_database, \
    get_column_header_dict, to_list_of_dicts


# Don't yell at me for this
def dd_dict():
    return defaultdict(dict)


def dd_int():
    return defaultdict(int)


def dd_dict_int():
    return defaultdict(dd_int)


def dd_dict_dict_int():
    return defaultdict(dd_dict_int)


def get_user_flows(database_name,
                db_config_file,
                input_table_name,
                user_accounts,
                user_aliases=None,
                output_filename='flows.csv',
                data_limit=100,
                time_interval='month',
                verbose=True):

    database, cursor = open_database(database_name, db_config_file)

    sql_statement = sql.SQL("""
    SELECT user_screen_name,user_name,user_description,user_created_ts,user_id,
    retweeted_status_user_screen_name,quoted_status_user_screen_name,in_reply_to_screen_name,
    created_ts
    -- complete_text
    FROM {input_table}
    """).format(input_table=sql.Identifier(input_table_name))
    
    for idx, user in enumerate(user_accounts):
        if idx == 0:
            sql_statement += sql.SQL("""
                WHERE retweeted_status_user_screen_name = {user_name}
                OR quoted_status_user_screen_name = {user_name}
                OR in_reply_to_screen_name = {user_name}""").format(user_name=sql.Literal(user))
        else:
            sql_statement += sql.SQL("""
                OR retweeted_status_user_screen_name = {user_name}
                OR quoted_status_user_screen_name = {user_name}
                OR in_reply_to_screen_name = {user_name}""").format(user_name=sql.Literal(user))

    if data_limit is not None:
        sql_statement += sql.SQL('LIMIT %s')

    cursor.execute(sql_statement, [data_limit])
    results = to_list_of_dicts(cursor)

    if verbose:
        for result in results:
            print(result)

    aggregate_dict = aggregate_flows(results, user_accounts)
    write_to_d3_sankey_csv(output_filename, aggregate_dict, 
            user_accounts, user_aliases)

    return


def aggregate_flows(flows, target_users):

    user_dict = dd_dict_dict_int()

    for flow in flows:
        user_id = flow['user_id']
        created_ts = flow['created_ts']

        if flow['quoted_status_user_screen_name'] is None or flow['quoted_status_user_screen_name'] not in target_users:
            if flow['in_reply_to_screen_name'] is None:
                interaction = flow['retweeted_status_user_screen_name']
            else:
                interaction = flow['in_reply_to_screen_name']
        else:
            interaction = flow['quoted_status_user_screen_name']

        month = f'{created_ts.month}_{created_ts.year}'
        user_dict[month][user_id][interaction] += 1

    for time_idx, (time, time_dict) in list(enumerate(user_dict.items())):
        for user, subuser_dict in list(time_dict.items()):
            target_counts = []
            for target_user in target_users:
                if target_user in subuser_dict:
                    target_counts += [subuser_dict[target_user]]
                else:
                    target_counts += [0]
            max_value = max(target_counts)
            max_indexes = [i for i, x in enumerate(target_counts) if x == max_value]
            target_choice = random.choice(max_indexes)
            user_dict[time][user] = target_users[target_choice]

    aggregate_dict = dd_dict_dict_int()

    # Very inefficient, will need to redo. Hard to keep this in my head..
    for time_idx, (time, time_dict) in enumerate(user_dict.items()):
        
        if time_idx == 0:
            continue  # Skip incomplete month
        if time_idx == 1:
            previous_time = time
            previous_time_dict = time_dict
            continue  # 

        for target_user in target_users + ['Entering']:
            for target_user2 in target_users + ['Exiting']:
                aggregate_dict[time][target_user][target_user2] = 0

        for user, current_choice in time_dict.items():
            if user in previous_time_dict.keys():
                previous_choice = previous_time_dict[user]
                aggregate_dict[time][previous_choice][current_choice] += 1
            else:
                aggregate_dict[time][current_choice]['Entering'] += 1

        for user, previous_choice in time_dict.items():
            aggregate_dict[previous_time]['Exiting'][previous_choice] += 1

        previous_time = time
        previous_time_dict = time_dict

    return aggregate_dict


def write_to_d3_sankey_csv(output_filename, aggregate_dict, target_users,
            user_aliases):

    """ This code has been hellish to write, and is unfinished.
    """
    colors = ['red', 'blue', 'green', 'orange']
    time_width = 250

    if user_aliases is None:
        color_dict = {user: colors[idx] for idx, user in enumerate(target_users)}
    else:
        color_dict = {user_aliases[user]: colors[idx] for idx, user in enumerate(user_aliases)}

    color_dict['Exiting'] = ''
    color_dict['Entering'] = ''

    time_intervals = []
    for year in ['2019', '2020']:
        for month in range(1, 13):
            time_intervals += [f'{month}_{year}']

    month_counter = 0
    with open(output_filename, 'w') as openfile:
        writer = csv.writer(openfile, delimiter=',')

        for time_idx, time_key in enumerate(time_intervals):
            if time_key in list(aggregate_dict.keys()):
                time_dict = aggregate_dict[time_key]

                month, year = str.split(time_key, '_')
                previous_month, previous_year = str.split(time_intervals[time_idx - 1], '_')
                next_month, next_year = str.split(time_intervals[time_idx + 1], '_')
                time_key = datetime.date(int(year), int(month), 1).strftime('%B, %Y')
                next_time_key = datetime.date(int(next_year), int(next_month), 1).strftime('%B, %Y')
                previous_time_key = datetime.date(int(previous_year), int(previous_month), 1).strftime('%B, %Y')

                for entering_key, exiting_dict in time_dict.items():
                    if user_aliases is not None and entering_key != 'Exiting' and entering_key != 'Entering':
                        entering_key = user_aliases[entering_key]
                    if entering_key == 'Exiting':
                        out_node = f'{next_time_key}, {entering_key}'
                    else:
                        out_node = f'{time_key}, {entering_key}'                
                    for exiting_key, value in exiting_dict.items():
                        if user_aliases is not None and exiting_key != 'Exiting' and exiting_key != 'Entering':
                            exiting_key = user_aliases[exiting_key]
                        if entering_key == 'Exiting':
                            in_node = f'{time_key}, {exiting_key}'
                        else:
                            in_node = f'{previous_time_key}, {exiting_key}'  

                        print(exiting_key)
                        if value != 0:
                            if 'Entering' in in_node:
                                displacement = (int(month) - 8) * time_width + 1
                            # elif 'Exiting' in out_node:
                            #     displacement = (int(month) - 6) * time_width + 1
                            else:
                                displacement = (int(month) - 7) * time_width + 1

                            # if 'July' in

                            writer.writerow([in_node, out_node, value, color_dict[exiting_key], displacement])

