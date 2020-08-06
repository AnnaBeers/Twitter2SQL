import lxml.etree as etree
import networkx as nx
import os
import pickle

from psycopg2 import sql
from pprint import pprint
from datetime import datetime, timezone, timedelta
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations

from twitter2sql.core.util import open_database, save_to_csv, \
        to_list_of_dicts, to_pandas, set_dict, int_dict, dict_dict
from twitter2sql.core import sql_statements


def generate_network_gefx(database_name,
                db_config_file,
                output_network_file,
                save_pkl=True,
                load_from_pkl=True,
                load_from_gexf=False,
                input_network_file=None,
                dict_pkl_file=None,
                users_pkl_file=None,
                mutual_pkl_file=None,
                table_name='tweets',
                connection_type='retweet',
                link_type='mutual',
                conditions=[],
                attributes=None,
                label='screen_name',
                connection_limit=10,
                network_pruning=10,
                itersize=1000,
                limit=None,
                mode='networkx',
                overwrite=False):

    # Type of tweet requested.
    if connection_type not in ['retweet', 'quote', 'reply', 'mention', 'all']:
        raise ValueError(f'connection_type must be retweet, quote, reply, mention, all, \
                not, {connection_type}')

    # This is for output types.
    if mode not in ['networkx', 'dynamic']:
        raise ValueError(f'connection_type must be networkx, dynamic, \
                not, {mode}')

    output_columns = set()
    if attributes is not None:
        output_columns += attributes
    output_columns.update(['user_id', 'user_name', 'user_screen_name'])

    tweet_type_condition = sql_statements.tweet_formats(connection_type)
    where_statement = sql_statements.format_conditions([tweet_type_condition] + conditions)

    # Wonder if this could be exported to a util function.
    if connection_type == 'retweet':
        output_columns.update(['retweeted_status_user_id', 'retweeted_status_user_screen_name'])
        connect_column = 'retweeted_status_user_id'
        connect_column_screen_name = 'retweeted_status_user_screen_name'
    elif connection_type == 'quote':
        output_columns.update(['quoted_status_user_id', 'quoted_status_user_screen_name'])
        connect_column = 'quoted_status_user_id'
        connect_column_screen_name = 'quoted_status_user_screen_name'
    elif connection_type == 'reply':
        output_columns.update(['in_reply_to_user_id', 'in_reply_to_user_screen_name'])
        connect_column = 'in_reply_to_user_id'
        connect_column_screen_name = 'in_reply_to_user_screen_name'
    elif connection_type == 'mention':
        raise NotImplementedError('Mentions not yet implemented')
    elif connection_type == 'all':
        output_columns += ['quoted_status_user_id', 'quoted_status_user_screen_name', 
                'retweeted_status_user_id', 'retweeted_status_user_screen_name',
                'in_reply_to_user_id', 'in_reply_to_user_screen_name']

    graph = None
    if not overwrite and load_from_gexf and os.path.exists(input_network_file):
        graph = nx.read_gexf(input_network_file)
    elif not overwrite and load_from_pkl and os.path.exists(dict_pkl_file) and os.path.exists(users_pkl_file):
        print('Loading input dict')
        with open(dict_pkl_file, 'rb') as openfile:
            connections_dict = pickle.load(openfile)
        print('Loading user dict')
        with open(users_pkl_file, 'rb') as openfile:
            user_dict = pickle.load(openfile)
        if mutual_pkl_file is not None and link_type == 'mutual':
            if os.path.exists(mutual_pkl_file):
                print('Loading mutual connections dict')
                with open(mutual_pkl_file, 'rb') as openfile:
                    mutual_dict = pickle.load(openfile)
            else:
                mutual_dict = None
        else:
            mutual_dict = None

    else:
        connections_dict, user_dict, = stream_connection_data(database_name, 
                db_config_file, output_network_file, output_columns,
                connect_column, connect_column_screen_name, where_statement,
                save_pkl, dict_pkl_file, users_pkl_file,
                table_name, connection_type, 
                attributes, label, itersize, 
                limit)

    if mode == 'networkx':
        if graph is None:
            graph = process_dicts_nx(connections_dict, user_dict, connect_column,
                connect_column_screen_name, connection_limit, link_type, mutual_dict)
            nx.write_gexf(graph, output_network_file)

        if network_pruning:

            components = list(nx.weakly_connected_components(graph))
            for component in components:
                if len(component) < network_pruning:
                    for node in component:
                        graph.remove_node(node)

            remove_count = 1
            while remove_count != 0:
                remove_count = 0
                for node in list(graph.nodes):
                    in_edges = len(graph.in_edges(node))
                    out_edges = len(graph.out_edges(node))
                    if out_edges < 3 and in_edges < 5:
                    # if in_edges < 5:
                        out_connects = list(graph.out_edges(node))
                        out_connects_check = []
                        remove = True
                        for out_node in out_connects:
                            out_node = out_node[0]
                            if len(graph.in_edges(out_node)) > 3:
                                remove = False
                        if remove:
                            remove_count += 1
                            graph.remove_node(node)

                print(f'Removed: {remove_count}')
                print(f'Total: {len(graph.nodes)}')

            output_network_file = output_network_file.replace('.gexf', f'_{network_pruning}.gexf')

            nx.write_gexf(graph, output_network_file)

    elif mode == '':
        network_data = process_dicts(connections_dict, user_dict, connect_column,
                connect_column_screen_name, connection_limit)

        create_gexf(network_data, output_network_file,
            id_col='user_id',
            edge_col='connect_id',
            label_col='user_screen_name',
            edge_label_col='connect_screen_name',
            weight_col='weight',
            dynamic=False,
            time_col='created_ts',
            attribute_dict={})


def stream_connection_data(database_name,
                db_config_file,
                output_gefx_file,
                output_columns,
                connect_column,
                connect_column_screen_name,
                where_statement,
                save_pkl=True,
                dict_pkl_file=None,
                users_pkl_file=None,
                table_name='tweets',
                connection_type='retweet',
                attributes=None,
                label='screen_name',
                itersize=1000,
                limit=None):

    if limit is None:
        limit_statement = sql.SQL('')
    else:
        limit_statement = sql.SQL(f'LIMIT {limit}')

    select_columns = sql.SQL(', ').join([sql.Identifier(item) for item in output_columns])

    database, cursor = open_database(database_name, db_config_file, 
            named_cursor='network_connections_retrieval', itersize=itersize)

    user_statement = sql.SQL("""
        SELECT {select}
        FROM {table_name}
        {where_statement}
        {limit_statement}
        """).format(table_name=sql.SQL(table_name),
                select=select_columns, where_statement=where_statement,
                limit_statement=limit_statement)
    cursor.execute(user_statement)
    
    connections_dict = defaultdict(dict_dict)
    username_dict = defaultdict(set_dict)

    count = 0
    progress_bar = tqdm()
    while True:
        result = cursor.fetchmany(cursor.itersize)
        if result:
            for item in result:
                item = dict(item)

                username_dict[item['user_id']]['screen_name'].add(item['user_screen_name'])

                if connection_type in ['reply', 'quote', 'retweet']:
                    if 'count' not in connections_dict[item['user_id']][item[connect_column]]:
                        connections_dict[item['user_id']][item[connect_column]]['count'] = 0
                    connections_dict[item['user_id']][item[connect_column]]['count'] += 1
                    username_dict[item[connect_column]]['screen_name'].add(item[connect_column_screen_name])
                elif connection_type == 'all':
                    pass

                if attributes is not None:
                    for attribute in attributes:
                        connections_dict[item['user_id']][item[connect_column]][attribute] = item[attribute]

            count += len(result)
            progress_bar.set_description(f"Iteration {count // itersize}, {count} rows retrieved.")
        else:
            cursor.close()
            break

    if save_pkl:
        with open(dict_pkl_file, 'wb') as openfile:
            pickle.dump(connections_dict, openfile)
        with open(users_pkl_file, 'wb') as openfile:
            pickle.dump(username_dict, openfile)

    return connections_dict, username_dict


def process_dicts(input_dict, user_dict, connect_column,
            connect_column_screen_name, connection_limit=20,
            connection_mode='direct'):

    if connection_mode == 'direct':
        output_data = []
        for connecting_user, connecting_dict in input_dict.items():
            for connected_user, connected_dict in connecting_dict.items():
                if connected_dict['count'] >= connection_limit:
                    data_dict = {key: value for key, value in connected_dict.items() if key != 'count'}
                    data_dict['user_id'] = connecting_user
                    data_dict['connect_id'] = connected_user
                    data_dict['user_screen_name'] = next(iter(user_dict[connecting_user]['screen_name']))
                    data_dict['connect_screen_name'] = next(iter(user_dict[connected_user]['screen_name']))
                    data_dict['weight'] = connected_dict['count']
                    output_data += [data_dict]
        
    elif connection_mode == 'mutual':

        raise NotImplementedError
        output_data = None

    return output_data


def process_dicts_nx(input_dict, user_dict, connect_column,
            connect_column_screen_name, connection_limit=20,
            connection_mode='mutual', mutual_dict=None):

    graph = nx.DiGraph()

    if connection_mode == 'direct':

        for connecting_user, connecting_dict in tqdm(input_dict.items()):
            for connected_user, connected_dict in connecting_dict.items():
                if connected_dict['count'] >= connection_limit:
                    if connecting_user == connected_user:
                        continue
                    graph.add_edges_from([(connecting_user, connected_user)])
                    # graph.add_weighted_edges_from([(connecting_user, connected_user, connected_dict['count'])])
                    for key, value in connected_dict.items():
                        if key != 'count':
                            graph[connecting_user][connected_user][key] = value
                    graph.add_node(connecting_user, label=next(iter(user_dict[connecting_user]['screen_name'])))
                    graph.add_node(connected_user, label=next(iter(user_dict[connected_user]['screen_name']))) 

    elif connection_mode == 'mutual':

        if mutual_dict is None:
            mutual_dict = defaultdict(int_dict)
            for connecting_user, connecting_dict in tqdm(input_dict.items()):
                connected_users = list(connecting_dict.keys())
                pairs = combinations(connected_users, 2)
                for pair in pairs:
                    if pair[1] in mutual_dict[pair[0]].keys():
                        pair = [pair[1], pair[0]]
                    mutual_dict[pair[0]][pair[1]] += 1

        for connecting_user, connecting_dict in tqdm(mutual_dict.items()):
            for connected_user, count in connecting_dict.items():
                if count >= connection_limit:
                    graph.add_edges_from([(connecting_user, connected_user)])
                    graph.add_node(connecting_user, label=next(iter(user_dict[connecting_user]['screen_name'])))
                    graph.add_node(connected_user, label=next(iter(user_dict[connected_user]['screen_name'])))

        pass

    return graph


def prune_data(output_data, network_size):

    return


def create_gexf(input_data, output_filename,
        id_col='user_id',
        edge_col='in_reply_to_user_id',
        label_col='user_screen_name',
        edge_label_col='in_reply_to_user_screen_name',
        weight_col=None,
        dynamic=False,
        time_col='created_ts',
        attribute_dict=None):
    attr_qname = etree.QName("http://www.w3.org/2001/XMLSchema-instance", "schemaLocation")

    gexf = etree.Element('gexf',
                         {attr_qname: 'http://www.gexf.net/1.3draft  http://www.gexf.net/1.3draft/gexf.xsd'},
                         nsmap={None: 'http://graphml.graphdrawing.org/xmlns/graphml'},
                         version='1.3')

    if dynamic:
        graph = etree.SubElement(gexf,
                                 'graph',
                                 defaultedgetype='directed',
                                 mode='dynamic',
                                 timeformat='datetime')
    else:
        graph = etree.SubElement(gexf,
                                 'graph',
                                 defaultedgetype='directed')

    attributes = etree.SubElement(graph, 'attributes', {'class': 'node', 'mode': 'static'})
    if attribute_dict is not None:
        for key, subdict in attribute_dict: 
            etree.SubElement(attributes, 'attribute', {'id': key, 'title': key['title'], 'type': key['type']})

    nodes = etree.SubElement(graph, 'nodes')
    edges = etree.SubElement(graph, 'edges')

    for item in tqdm(input_data):

        item = {key: str(value) for key, value in item.items()}

        node, edge_node = add_edge(item, nodes, edges, 
                id_col, edge_col, label_col, edge_label_col, 
                weight_col, dynamic, time_col)

        if attribute_dict is not None:
            add_node_attributes(item, node, attribute_dict)
            add_node_attributes(item, edge_node, attribute_dict)

    with open(output_filename, 'w', encoding='utf-8')as f:
        f.write(etree.tostring(gexf, encoding='utf8', method='xml').decode('utf-8'))

    return output_filename


def add_node_attributes(item, node, attribute_dict):
    attvalues = etree.SubElement(node, 'attvalues')
    for key in attribute_dict:
        if key in item:
            etree.SubElement(attvalues,
                             'attvalue',
                             {'for': key,
                             'value': str(item[key])})


def add_edge(item, nodes, edges, 
            id_col, edge_col, label_col, edge_label_col, weight_col=None,
            dynamic=False, time_col=None):
    node = etree.SubElement(nodes,
                        'node',
                        id=item[id_col],
                        Label=item[label_col])
    edge = etree.SubElement(edges,
                 'edge',
                 {'id': item[id_col],
                 'source': item[id_col],
                 'target': item[edge_col]})

    if weight_col is not None:
        edge.set("weight", str(item[weight_col]))

    edge_node = etree.SubElement(nodes,
                            'node',
                            id=item[edge_col],
                            Label=item[edge_label_col])

    if dynamic:
        node.set('start', item[time_col].isoformat(timespec='seconds'))
        node.set('end', (item[time_col] + timedelta(seconds=1)).isoformat(timespec='seconds'))
        edge_node.set('start', item[time_col].isoformat(timespec='seconds'))
        edge_node.set('end', (item[time_col] + timedelta(seconds=1)).isoformat(timespec='seconds'))

    return node, edge_node


if __name__ == '__main__':
    pass