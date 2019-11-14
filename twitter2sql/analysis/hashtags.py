import psycopg2

from psycopg2 import sql
from pprint import pprint

from twitter2sql.core import sql_statements
from twitter2sql.core.util import clean, c, get_last_modified, \
    within_time_bounds, open_database, close_database, \
    get_column_header_dict, to_list_of_dicts

"""
How to create a hashtag table in postgres
CREATE TABLE hashtags as
SELECT  "id" as tweetid, t->>'text' as hashtag
FROM    tweets j
CROSS JOIN
        json_array_elements("entities"->'hashtags') t
"""


def hashtag_sql_statement(input_table_name, output_table_name, hashtag_col, 
                            select):

    create_statement = sql.SQL("""CREATE TABLE IF NOT EXISTS {} AS
                            """.format(output_table_name))

    create_statement += sql.SQL("""
        SELECT {select}, hashtag_obj->>'text' AS {hashtag}
        FROM {table}
        CROSS JOIN json_array_elements("entities"->'hashtags') hashtag_obj
        """).format(select=sql.Identifier(select),
                        table=sql.Identifier(input_table_name),
                        hashtag=sql.Identifier(hashtag_col))

    return create_statement


def generate_hashtag_table(database_name,
                db_config_file,
                input_table_name,
                output_table_name,
                hashtag_col,
                select_col,
                overwrite=False):

    database, cursor = open_database(database_name, db_config_file)

    if overwrite:
        cursor.execute(sql_statements.drop_table_statement(output_table_name))

    hashtag_create_statement = hashtag_sql_statement(input_table_name, 
                                                    output_table_name,
                                                    hashtag_col,
                                                    select_col)

    cursor.execute(hashtag_create_statement)
    database.commit()

    return


def append_hashtag_clusters(database_name,
                db_config_file,
                table_name)

    return


if __name__ == '__main__':
    pass