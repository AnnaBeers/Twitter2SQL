""" TODO: Build utilities for sharing cursors between functions.
"""

import datetime

from psycopg2 import sql
from collections import OrderedDict

from twitter2sql.core.util import open_database, save_to_csv, to_list_of_dicts


def aggregate_by_time(database_name,
                        db_config_file,
                        column_name='example',
                        num_returned=10000,
                        table_name='table',
                        return_headers=None,
                        distinct=None,
                        output_filename=None,
                        output_column_headers=None):

    return


def filter_by_time(database_name,
                        db_config_file,
                        time_column='example',
                        num_returned=10000,
                        table_name='table',
                        return_headers=None,
                        distinct=None,
                        output_filename=None,
                        output_column_headers=None):

    database, cursor = open_database(database_name, db_config_file)

    cursor.execute(sql.SQL("""
        SELECT *
        FROM {}
        WHERE time_column BETWEEN %s and %s
        LIMIT %s;
        """).format(sql.Identifier(table_name), sql.Identifier(column_name)), [start_date, end_date])

    results = cursor.fetchall()
    dict_result = []
    for row in results:
        dict_result.append(dict(row))

    results = remove_duplicates(dict_result, limit=100)

    if output_filename is not None:
        save_to_csv(results, output_filename, output_column_headers)

    return results


class Command(object):

    def __init__(self, verbose=False):

        self.verbose = verbose

        return

    def execute_sql(self):

        return


def aggregate(database_name,
                db_config_file,
                aggregate_column='example',
                output_columns='example',
                table_name='table',
                count_column_name='total_tweets',
                num_returned=1000,
                return_headers=None,
                output_filename=None,
                output_column_headers=None,
                verbose=False):

    database, cursor = open_database(database_name, db_config_file)

    # Max is a bit shady here.
    output_columns_sql = sql.SQL(',').join([sql.SQL("MAX({output}) as {output}").format(output=sql.Identifier(output)) for output in output_columns])
    sql_statement = sql.SQL("""
        SELECT {agg}, COUNT({count}) as {count_name},
        {outputs}
        FROM {table}
        GROUP BY {agg}
        ORDER BY {count_name} DESC
        LIMIT %s;
        """).format(agg=sql.Identifier(aggregate_column),
                    count=sql.Identifier(aggregate_column),
                    count_name=sql.Identifier(count_column_name),
                    table=sql.Identifier(table_name),
                    outputs=output_columns_sql)
    cursor.execute(sql_statement, [num_returned])  

    results = to_list_of_dicts(cursor)

    if verbose:
        for result in results:
            print(result)

    if output_filename is not None:
        save_to_csv(results, output_filename, output_column_headers)

    return


def grab_top(database_name=None,
                        db_config_file=None,
                        cursor=None,
                        column_name='example',
                        num_returned=100,
                        table_name='table',
                        return_headers=None,
                        distinct=None,
                        output_filename=None,
                        output_column_headers=None):

    if cursor is None:
        database, cursor = open_database(database_name, db_config_file)
    else:
        return

    if distinct is None:
        sql_statement = sql.SQL("""
            SELECT user_screen_name,user_name,user_description,user_created_ts,user_followers_count,user_id,created_at,complete_text
            FROM (SELECT DISTINCT ON (user_id) user_screen_name,user_name,user_description,user_created_ts,user_followers_count,user_id,created_at,complete_text 
            FROM {} WHERE lang='en') as sub_table
            ORDER BY {} DESC
            LIMIT %s;
            """).format(sql.Identifier(table_name), sql.Identifier(column_name))
        cursor.execute(sql_statement, [num_returned])
    else:
        # Currently non-functional.
        cursor.execute(sql.SQL("""
            SELECT * FROM (
                SELECT DISTINCT ON {} *
            FROM {}
            ORDER BY {} DESC
            LIMIT %s;
            """).format(sql.Identifier(distinct),
                sql.Identifier(table_name), 
                sql.Identifier(column_name)), 
                [num_returned])

    results = to_list_of_dicts(cursor)
    # results = remove_duplicates(results, limit=100)

    if output_filename is not None:
        save_to_csv(results, output_filename, output_column_headers)

    return results


def remove_duplicates(rows, duplicate_key='user_id', sort_key='user_followers_count', limit=None):

    """ Removes duplicates in a list, can return only top 'limit' results.
        Placeholder function until I find how to do this in SQL.
        Assumes list is sorted.
        This is like the worst function I have ever made.
    """

    output_rows = []
    key_dict = OrderedDict()

    for idx, item in enumerate(rows):

        item_key = item[duplicate_key]

        # Very difficult to understand code.
        if item_key not in key_dict:
            key_dict[item_key] = item
        else:
            if key_dict[item_key][sort_key] < item[sort_key]:
                key_dict[item_key] = item

        if limit is not None:
            if len(key_dict) == limit:
                break

    print('Removed duplicates from..', idx)

    for key, value in key_dict.items():
        output_rows += [value]

    return output_rows


if __name__ == '__main__':

    grab_top()