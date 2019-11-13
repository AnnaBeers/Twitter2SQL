import time
import os
import psycopg2
import csv

from datetime import timedelta
from datetime import datetime
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


def open_database(database_name, 
                    db_config_file, 
                    create_new_db=False, 
                    owner='example',
                    admins=[]):

    # Parse the database credentials out of the file
    database_config = {"database": database_name}
    for line in open(db_config_file).readlines():
        key, value = line.strip().split("=")
        database_config[key] = value

    if create_new_db:
        create_statement = """CREATE DATABASE {db}
            WITH
            OWNER = {owner}
            ENCODING = 'UTF8'
            LC_COLLATE = 'en_US.UTF-8'
            LC_CTYPE = 'en_US.UTF-8'
            TABLESPACE = pg_default
            CONNECTION LIMIT = -1;
            """.format(db=database_name, owner=owner)
        public_permissions = """GRANT TEMPORARY, CONNECT ON DATABASE {db} TO PUBLIC;""".format(db=database_name)
        owner_permissions = """GRANT ALL ON DATABASE {db} TO {user};""".format(db=database_name, user=owner)
        
        admin_permissions = []
        for admin in admins:
            admin_permissions += ['\nGRANT TEMPORARY ON DATABASE {db} to {user}'.format(db=database_name, user=admin)]

        all_commands = [create_statement] + [public_permissions] + [owner_permissions] + admin_permissions

        create_database_config = database_config
        create_database_config['database'] = 'postgres'
        database = psycopg2.connect(**database_config)
        database.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = database.cursor(cursor_factory=psycopg2.extras.DictCursor)

        for command in all_commands:
            cursor.execute(command)
            database.commit()
        cursor.close()
        database.close()

    # Connect to the database and get a cursor object
    database = psycopg2.connect(**database_config)
    cursor = database.cursor(cursor_factory=psycopg2.extras.DictCursor)

    return database, cursor


def get_column_header_dict(input_column_csv):

    column_header_dict = {}
    with open(input_column_csv, 'r') as readfile:
        reader = csv.reader(readfile, delimiter=',')
        next(reader)  # This line skips the header row.

        for row in reader:
            column_header_dict[row[0]] = {'type': row[2], 'json_fieldname': row[1], 'clean': row[4], 'instructions': row[5]}
            if column_header_dict[row[0]]['clean'] == 'TRUE':
                column_header_dict[row[0]]['clean'] = True
            else:
                column_header_dict[row[0]]['clean'] = False

    return column_header_dict


def close_database(cursor, database, commit=True):
    # Close everything
    cursor.close()

    if commit:
        database.commit()

    database.close()


def clean(s):

    # Fix bytes/str mixing from earlier in code:
    if type(s) is bytes:
        s = s.decode('utf-8')

    # Replace weird characters that make Postgres unhappy
    return s.replace("\x00", "") if s else None


def c(u):
    # Encode unicode so it plays nice with the string formatting
    return u.encode('utf8')


def get_last_modified(json_file):
    return os.path.getmtime(json_file)


def within_time_bounds(json_file, start_time, end_time):
    json_modified_time = get_last_modified(json_file)
    return (json_modified_time >= time.mktime(start_time.timetuple())) and (json_modified_time <= (time.mktime(end_time.timetuple()) + timedelta(days=1).total_seconds()))


def save_to_csv(rows, output_filename, column_headers=None):

    with open(output_filename, 'w') as outfile:
        writer = csv.writer(outfile, delimiter=',')

        if column_headers is None:
            writer.writerow(rows[0].keys())
        else:
            writer.writerow(column_headers)

        for item in rows:

            if column_headers is None:
                # This might not work if dictionaries don't pull out keys in same order.
                writer.writerow(item.values())
            else:
                output_row = [item[column] for column in column_headers]
                writer.writerow(output_row)

    return


def load_from_csv(input_csv, time_columns=[]):

    with open(input_csv, 'r') as readfile:
        reader = csv.reader(readfile, delimiter=',')

        output_dict_list = []
        header = next(reader)

        for row in reader:
            output_dict = {}
            for idx, item in enumerate(row):
                # Time conversion is a little inefficient. But who cares!
                if header[idx] in time_columns:
                    item = datetime.strptime(item, "%Y-%m-%d %H:%M:%S")    
                output_dict[header[idx]] = item
            output_dict_list += [output_dict]

    return output_dict_list


def list_on_key(dict_list, key):

    """ Is there a one-liner for this?
    """

    return_list = []
    for sub_dict in dict_list:
        return_list += [sub_dict[key]]

    return return_list


def extract_entity_to_column():

    return


def to_list_of_dicts(cursor):

    results = cursor.fetchall()
    dict_result = []
    for row in results:
        dict_result.append(dict(row))

    return dict_result