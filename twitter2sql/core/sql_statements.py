""" This code is stupid and gross but it's probably the least complicated way without adding extra libraries
"""

import csv

from psycopg2 import sql


class sql_statement(object):

    def __init__(self):

        """ There's some repeated code in these sql_statements. Think about
            putting into a class.
        """

        return


def not_null_statement(table_name, where_col, select=None, distinct=None):

    """ Requires you to fill in the dates in a subsequent step.
        Think about how to do this in a simple way.
    """

    if select is None:
        select = sql.Identifier('*')
    elif type(select) is list:
        select = sql.SQL(', ').join([sql.Identifier(item) for item in select])
    else:
        select = sql.Identifier(select)
    
    if distinct is None:
        distinct = sql.SQL('')
    else:
        distinct = sql.SQL("""DISTINCT ON ({})""").format(sql.Identifier(distinct))

    sql_statement = sql.SQL("""
        SELECT {distinct} {select}
        FROM {table}
        WHERE {where_col} IS NOT NULL;
        """).format(select=select, table=sql.Identifier(table_name), where_col=sql.Identifier(where_col), distinct=distinct)

    return sql_statement


def in_statement(table_name, where_col, values, select=None, distinct=None):

    """ Requires you to fill in the dates in a subsequent step.
        Think about how to do this in a simple way.
    """

    if select is None:
        select = sql.Identifier('*')
    elif type(select) is list:
        select = sql.SQL(', ').join([sql.Identifier(item) for item in select])
    else:
        select = sql.Identifier(select)
    
    if distinct is None:
        distinct = sql.SQL('')
    else:
        distinct = sql.SQL("""DISTINCT ON ({})""").format(sql.Identifier(distinct))

    # This is a bit suspect.
    value_string = ''
    for item in values:
        value_string += '%s,'
    value_string = sql.SQL(value_string[:-1])

    sql_statement = sql.SQL("""
        SELECT {distinct} {select}
        FROM {table}
        WHERE {where_col} in ({values});
        """).format(select=select, table=sql.Identifier(table_name), 
                        values=value_string, where_col=sql.Identifier(where_col), distinct=distinct)

    return sql_statement


def filter_date_statement(table_name, date_column='created_ts', select=None, distinct=None):

    """ Requires you to fill in the dates in a subsequent step.
        Think about how to do this in a simple way.
    """

    if select is None:
        select = '*'
    
    if distinct is None:
        distinct = ''
    else:
        distinct = sql.SQL("""DISTINCT ON ({})""").format(sql.Identifier(distinct))

    sql_statement = sql.SQL("""
        SELECT {distinct} {select}
        FROM {table}
        WHERE {date} BETWEEN %s AND %s
        """).format(select=sql.Identifier(select), table=sql.Identifier(table_name), 
                        date=sql.Identifier(date_column), distinct=distinct)

    return sql_statement


def filter_statement():

    """Coming soon!
    """

    return


def table_permission_statement(table_name, admins):

    """ Grants all permissions to the list of users in admins.
    """

    for admin in admins:
        sql_statement = """GRANT ALL ON TABLE public.{table} TO {user};""".format(table=table_name, user=admin)

    return sql_statement


def create_table_statement(input_schema, table_name):

    """ A generic table create. Schemes can be either Python 
    dicts or csv files.
    """

    create_statement = "CREATE TABLE IF NOT EXISTS {} (".format(table_name)
    
    if type(input_schema) is str:

        with open(input_schema, 'r') as readfile:
            reader = csv.reader(readfile, delimiter=',')
            next(reader)  # This line skips the header row.

            for row in reader:
                create_statement += ' '.join([row[0]] + row[2:4]) + ','

    elif type(input_schema) is dict:

        for column_header, data_type in input_schema.items():
            create_statement += ' '.join([column_header] + data_type) + ','

    else:
        print()

    create_statement = create_statement[:-1] + ')'  # Replace final comma with )
    create_statement += ";"

    return create_statement


def insert_statement(input_schema, table_name):

    """ A generic data insert. Schemes can be either Python 
    dicts or csv files.
    """

    insert_statement = "INSERT INTO {} (".format(table_name)

    if type(input_schema) is str:

        with open(input_schema, 'r') as readfile:
            reader = csv.reader(readfile, delimiter=',')
            next(reader)  # This line skips the header row.

            value_num = 0

            for row in reader:
                insert_statement += row[0] + ','
                value_num += 1

    elif type(input_schema) is dict:

        value_num = len(input_schema)
        for column_header, data_type in input_schema.items():
            insert_statement += column_header + ','

        return

    else:
        print()

    insert_statement = insert_statement[:-1] + ") VALUES ("  # Replace final comma with )
    for i in range(value_num):
        insert_statement += '%s,'
    insert_statement = insert_statement[:-1] + ')'  # Replace final comma with )

    return insert_statement