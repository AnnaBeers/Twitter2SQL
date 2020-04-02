""" Scripts for creating visualizations from PostgreSQL
    databases.
"""

import altair as alt

from psycopg2 import sql
from pprint import pprint

from twitter2sql.core.util import open_database, save_to_csv, to_list_of_dicts, to_pandas


def timeline_chart(database_name,
                db_config_file,
                tweet_table,
                start_date,
                end_date,
                output_filename):

    database, cursor = open_database(database_name, db_config_file)

    time_aggregate_statement = sql.SQL("""SELECT 
    date_trunc('day', created_ts) AS "date" , count(*) AS "tweet_total"
    FROM {tweet_table}
    WHERE created_ts >= %s
    AND created_ts <= %s
    GROUP BY 1 
    ORDER BY 1;
        """).format(tweet_table=sql.Identifier(tweet_table))
    cursor.execute(time_aggregate_statement, [start_date, end_date])  

    results = to_pandas(cursor)

    chart = alt.Chart(results).mark_line().encode(
        x='date',
        y='tweet_total',
        # color='symbol'
    )

    chart.save(output_filename)

    return


if __name__ == '__main__':

    pass
