import psycopg2

from psycopg2 import sql
from pprint import pprint

from twitter2sql.core import sql_statements
from twitter2sql.core.util import clean, c, get_last_modified, \
    within_time_bounds, open_database, close_database, \
    get_column_header_dict, to_list_of_dicts


def append_hashtag_clusters(database_name,
                db_config_file,
                hashtag_table_name,
                hashtag_cluster_csv,
                hashtag_col='hashtags',
                cluster_col='clusters'):

    database, cursor = open_database(database_name, db_config_file)

    create_col_statement = sql.SQL("""ALTER TABLE {table}\n
        ADD COLUMN IF NOT EXISTS {cluster_col} VARCHAR;""").format(
        table=sql.SQL(hashtag_table_name),
        cluster_col=sql.SQL(cluster_col))
    cursor.execute(create_col_statement)

    classify_statement, classify_values = sql_statements.category_statement(
        hashtag_table_name,
        hashtag_cluster_csv,
        hashtag_col,
        cluster_col)

    cursor.execute(classify_statement, classify_values)
    database.commit()
    

if __name__ == '__main__':

    pass