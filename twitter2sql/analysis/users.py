import psycopg2
import tweepy

from tweepy.auth import OAuthHandler
from psycopg2 import sql
from pprint import pprint

from twitter2sql.core import sql_statements
from twitter2sql.core.util import clean, c, get_last_modified, \
    within_time_bounds, open_database, close_database, \
    get_column_header_dict, to_list_of_dicts


def users_sql_statement(input_table_name, output_table_name, 
                        id_col, event_split_date=None):

    create_statement = sql.SQL("""CREATE TABLE IF NOT EXISTS {} AS
                            """.format(output_table_name))

    """
    SELECT created_at, sum(count(email)) OVER (ORDER BY created_at)
    FROM (
        SELECT DISTINCT ON (email) created_at, email
        FROM subscriptions ORDER BY email, created_at
    ) AS subq
    GROUP BY created_at;
    """

    create_statement += sql.SQL("""
        SELECT DISTINCT ON (user_id) 
        user_id,
        user_followers_count as last_user_followers_count,
        user_friends_count as last_user_friends_count,
        user_statuses_count as last_user_statuses_count,
        user_favourites_count as last_user_favourites_count,
        user_geo_enabled,user_time_zone,
        user_description as last_user_description,
        user_name as last_user_name,
        user_screen_name as last_user_screename,
        user_url as last_user_url,
        user_created_ts as user_created_ts,
        created_ts as last_created_ts
        FROM {table}
        ORDER BY user_id, created_ts DESC
        """).format(table=sql.Identifier(input_table_name),
                        id_col=sql.Identifier(id_col))

    return create_statement


def generate_users_table(database_name,
                db_config_file,
                input_table_name,
                user_table_name,
                user_id_col,
                admins,
                event_split_date=None,
                overwrite=False):

    database, cursor = open_database(database_name, db_config_file)

    if overwrite:
        cursor.execute(sql_statements.drop_table_statement(user_table_name))

    users_create_statement = users_sql_statement(input_table_name, 
                                                    user_table_name,
                                                    user_id_col,
                                                    event_split_date)

    cursor.execute(users_create_statement)
    database.commit()

    # Add admins to the table.
    admin_add_statement = sql_statements.table_permission_statement(
        user_table_name, 
        admins)
    cursor.execute(admin_add_statement)
    database.commit()

    return


def execute_user_sql(statement, column_name, dtype, cursor, database, message,
                    user_table_name='users', tweet_table_name='tweets'):

    create_col_statement = sql_statements.create_col_statement(user_table_name,
        column_name, dtype=dtype)
    cursor.execute(create_col_statement)    

    statement = statement.format(
        user_table=sql.Identifier(user_table_name),
        tweet_table=sql.Identifier(tweet_table_name),
        column_name=sql.Identifier(column_name))

    pprint("Updating {}..".format(message))
    cursor.execute(statement)
    database.commit()

    return


def generate_user_statistics(database_name,
                db_config_file,
                tweet_table_name,
                user_table_name,
                overwrite=False):

    database, cursor = open_database(database_name, db_config_file)
    
    # If statements are for testing, remove or make parameter eventually.

    # Total Posts
    if True:
        total_post_statement = sql.SQL("""UPDATE {user_table}\n
            SET {column_name} = temp.idcount
            FROM
            (SELECT COUNT(user_id) as idcount, user_id FROM {tweet_table} 
            GROUP BY user_id) AS temp
            WHERE {user_table}.user_id = temp.user_id
            """)

        execute_user_sql(total_post_statement, 'total_database_tweets', 'INT', 
            cursor, database, 'total post count')

    # First Post, Statuses Count
    if True:
        first_post_statement = sql.SQL("""UPDATE {user_table}\n
            SET first_created_ts = temp.first_post
            FROM
            (SELECT user_id, min(created_ts) first_post
                FROM {tweet_table}
                GROUP BY user_id) AS temp
            WHERE {user_table}.user_id = temp.user_id
            """)

        execute_user_sql(first_post_statement, 'first_created_ts', 'TIMESTAMP', 
            cursor, database, 'first post')

        first_post_statement = sql.SQL("""UPDATE {user_table}\n
            SET first_status_count = temp.first_status
            FROM
            (SELECT DISTINCT ON (user_id) user_id,
            user_statuses_count as first_status, user_created_ts
            FROM {tweet_table}
            ORDER BY user_id, user_created_ts ASC) AS temp
            WHERE {user_table}.user_id = temp.user_id
            """)

        execute_user_sql(first_post_statement, 'first_status_count', 'INT', 
            cursor, database, 'first status count')

    # Time Differences
    if True:
        date_range_statement = sql.SQL("""UPDATE {user_table}\n
            SET database_active_days = extract(day 
            FROM last_created_ts - first_created_ts)
            """)

        execute_user_sql(date_range_statement, 'database_active_days', 'INT', 
            cursor, database, 'active days in database')

        date_range_statement = sql.SQL("""UPDATE {user_table}\n
            SET previous_active_days = extract(day 
            FROM first_created_ts - user_created_ts)
            """)

        execute_user_sql(date_range_statement, 'previous_active_days', 'INT', 
            cursor, database, 'previously active days')

    # Rates
    if True:
        rate_statement = sql.SQL("""UPDATE {user_table}\n
            SET database_posting_rate = cast(total_database_tweets as numeric)
            / NULLIF(database_active_days,0)
            """)

        execute_user_sql(rate_statement, 'database_posting_rate', 'NUMERIC', 
            cursor, database, 'database posting rate')

        rate_statement = sql.SQL("""UPDATE {user_table}\n
            SET previous_posting_rate = cast(first_status_count as numeric)
            / NULLIF(previous_active_days,0)
            """)

        execute_user_sql(rate_statement, 'previous_posting_rate', 'NUMERIC', 
            cursor, database, 'previous rate statement')        

        rate_statement = sql.SQL("""UPDATE {user_table}\n
            SET previous_current_posting_rate_ratio = database_posting_rate
            / NULLIF(previous_posting_rate,0)
            """)

        execute_user_sql(rate_statement, 'previous_current_posting_rate_ratio', 
            'NUMERIC', cursor, database, 'posting rate ratio statement')   

    return


def generate_suspended_users(database_name,
                db_config_file,
                tweet_table_name,
                user_table_name,
                total_posts_col,
                overwrite=False):

    """ This might load too much into memory for large databases.
    """

    return


def pull_user_mentions(database_name,
                db_config_file,
                tweet_table_name):

    return


if __name__ == '__main__':
    pass