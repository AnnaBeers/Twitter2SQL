""" Scripts for analyzing coded data stored in Google Sheets.
    Currently mostly works for one application.
"""

import os
import csv
import random

from pprint import pprint
from collections import Counter, defaultdict, namedtuple
from itertools import combinations

import xlsxwriter
import pandas as pd
import numpy as np

from sklearn.metrics import cohen_kappa_score, confusion_matrix
from tqdm import tqdm


def analyze_codes(
        input_data,
        coders,
        output_directory,
        suffix='',
        multi_select_codes=None,
        lead_code=None,
        code_hierarchy=None,
        exclude_codes=None,
        exclude_values=None,
        arb_cols=None,
        code_groups=None,
        max_raters=2,
        aggregate_stats=True,
        confusion_matrices=True,
        pairwise_stats=True,
        arb=True,
        individual_stats=True,
        discussion=True,
        verbose=True,
        exclude=None):

    """ Input xlsx is expected to have the following format:
        1. A 'Codebook' sheet with vertical cols of titled codes.
        2. Coding sheets per coder titled 'Tweets_{coder}' 
        3. Any number of unrelated sheets.

        Each coding sheet is expected to have, in order:
        1. Any number of data cols
        2. TRUE/FALSE coder cols = to # of coders.
        3. Code cols

    """

    if exclude_codes is None:
        exclude_codes = ['Notes']
    if exclude_values is None:
        exclude_values = ['Unclear']

    code_level_stats = os.path.join(output_directory, f'Code_Level_Stats{suffix}.csv')
    confusion_matrix_stats = os.path.join(output_directory, f'Confusion_Matrices{suffix}.csv')
    confusion_matrix_image_dir = os.path.join(output_directory, f'Confusion_Matrices{suffix}')
    individual_csv = os.path.join(output_directory, f'Individual_Statistics{suffix}.csv')
    arb_csv = os.path.join(output_directory, f'Arbitration{suffix}.csv')
    output_xlsx = os.path.join(output_directory, f'Coding_Analysis{suffix}.xlsx')
    discussion_xlsx = os.path.join(output_directory, f'Discussion{suffix}.xlsx')

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    if not os.path.exists(confusion_matrix_image_dir):
        os.mkdir(confusion_matrix_image_dir)

    file_type = os.path.splitext(input_data)[1]

    """ 1. Load data
    """

    # Maybe support for non-Sheets coding later?
    if file_type not in ['.xlsx']:
        raise NotImplementedError

    if file_type == '.xlsx':

        raw_xlsx = pd.ExcelFile(input_data)

        data_dict = {}
        for coder in tqdm(coders):
            for name in raw_xlsx.sheet_names:
                if coder in name:
                    df = raw_xlsx.parse(name, keep_default_na=False, na_values=[''])
                    df = df.dropna(subset=['Tweet'])
                    data_dict[coder] = df

        code_dict = {}
        codebook = raw_xlsx.parse('Codebook', keep_default_na=False, na_values=[''])
        for col in list(codebook):
            code_list = list(codebook[col].dropna().astype(str))
            code_list = [x for x in code_list if x not in exclude_values]
            code_dict[col] = code_list

    """ 2. Extract codes and identify coders. Not great code, but extracts codes according to the
        order specified above.
    """

    codes_only = []
    past_coders = False
    for colname in list(data_dict[coders[0]]):
        if colname in coders:
            past_coders = True
        if past_coders:
            if colname not in coders and colname != exclude:
                codes_only += [colname]
    analysis_codes = [x for x in codes_only if x not in exclude_codes]

    if lead_code is None:
        lead_code = analysis_codes[0]
    print(lead_code)

    if max_raters is None:
        max_raters = len(coders)

    """ 3. Write out data.
    """

    with open(code_level_stats, 'w') as codefile, \
            open(confusion_matrix_stats, 'w') as confusionfile, \
            open(arb_csv, 'w') as arbfile, \
            open(individual_csv, 'w') as indivfile, \
            xlsxwriter.Workbook(output_xlsx) as workbook, \
            xlsxwriter.Workbook(discussion_xlsx) as dworkbook:

        # First figure out filewriting (ugh)...
        stats_sheet, confusion_sheet, arb_sheets, individual_sheet, \
            writer, confusion_writer, arb_writer, individual_writer, \
            code_header, individual_header = format_filewriters(
                workbook, arb_cols, coders + ['Discussion'],
                codes_only, data_dict, codefile, arbfile, confusionfile,
                indivfile)

        confusion_rownum, individual_rownum, stats_rownum = 0, 1, 1
        arb_rownums = {coder: 1 for coder in coders}

        # Calculate individual ratios
        if individual_stats:
            process_individual_stats(
                individual_writer, individual_sheet, individual_rownum,
                data_dict, lead_code, analysis_codes, multi_select_codes,
                individual_header)

        # Get scores for the aggregate data first..
        if aggregate_stats:
            coder1, coder2 = '0', '1'

            combined = process_aggregate_codesheet(data_dict, analysis_codes, lead_code)
            output_row = {'Combined Pair': 'All'}

            stats_rownum, confusion_rownum = write_data(
                output_row, combined, coder1, coder2, code_dict,
                code_hierarchy, multi_select_codes, code_groups,
                lead_code, exclude_values, confusion_matrices,
                confusion_rownum, confusion_writer, confusion_sheet,
                analysis_codes, stats_rownum, stats_sheet, code_header,
                writer)

        # Calculate scores for each coding pair.
        if pairwise_stats:
            for idx, (coder1, coder2) in enumerate(combinations(coders, 2)):

                combined = process_pair_codesheet(
                    data_dict, coders, analysis_codes, coder1, coder2, lead_code)
                output_row = {
                    'Combined Pair': f'{coder1}_{coder2}',
                    'Pair 1': coder1, 'Pair 2': coder2}

                stats_rownum, confusion_rownum = write_data(
                    output_row, combined, coder1, coder2, code_dict,
                    code_hierarchy, multi_select_codes, code_groups,
                    lead_code, exclude_values, False,
                    confusion_rownum, confusion_writer, confusion_sheet,
                    analysis_codes, stats_rownum, stats_sheet, code_header,
                    writer)

        # Then write out rows for arb
        if arb:
            write_arb(
                arb_writer, data_dict, lead_code, arb_cols,
                codes_only, arb_sheets, arb_rownums)

        if discussion:
            write_discussion(dworkbook, data_dict, lead_code, codes_only, coders, analysis_codes)

    return output_xlsx


def write_data(
        output_row, combined, coder1, coder2, code_dict,
        code_hierarchy, multi_select_codes, code_groups,
        lead_code, exclude_values, confusion_matrices,
        confusion_rownum, confusion_writer, confusion_sheet,
        analysis_codes, stats_rownum, stats_sheet, code_header,
        writer):

    for code in tqdm(analysis_codes):
        output_row['Code'] = code
        output_row = calculate_all_scores(
            output_row, combined, coder1, coder2, code_dict,
            code, code_hierarchy, multi_select_codes, code_groups,
            lead_code, exclude_values)

        if confusion_matrices:
            confusion_rownum = write_confusion(
                combined, confusion_writer, confusion_sheet, confusion_rownum,
                code, coder1, coder2, code_dict, multi_select_codes)

        stats_rownum = write_xlsx_row(
            stats_sheet, output_row,
            stats_rownum, code_header)
        writer.writerow(output_row)

    return stats_rownum, confusion_rownum


def write_xlsx_row(sheet, row, rownum, code_header=None):

    if isinstance(row, dict):
        row = [row[x] if x in row else '' for x in code_header]

    sheet.write_row(rownum, 0, row)
    rownum += 1

    return rownum


def format_filewriters(
        workbook, arb_cols,
        coders, codes_only, data_dict,
        codefile, arbfile, confusionfile, indivfile):

    # XLSX Sheet Creation

    # Agreement Statistics
    code_header = ['Combined Pair', 'Pair 1', 'Pair 2', 'Code']

    extra_measures = ['', 'Conditional_', 'Partial_', 'Grouped_']
    for extra in extra_measures:
        code_header += [f'{extra}N', f'{extra}Agreement', f'{extra}Cohen_Kappa']

    stats_sheet = workbook.add_worksheet('Agreement_Statistics')
    stats_sheet.write_row(0, 0, code_header)

    # Individual Statistics
    individual_header = ['Coder', 'Code', 'Value', 'N', 'Percent', 'Multi_N', 'Multi_Percent']
    individual_sheet = workbook.add_worksheet('Individual_Statistics')
    individual_sheet.write_row(0, 0, individual_header)

    # Confusion Matrices
    confusion_sheet = workbook.add_worksheet('Confusion_Statistics')

    # Arbitration
    if arb_cols is None:
        arb_header = list(data_dict[coders[0]])
    else:
        arb_header = [x for x in list(data_dict[coders[0]]) if x in arb_cols]

    arb_header = arb_header + ['Coder', 'Arbitrate?'] + codes_only
    arb_sheets = {}
    for coder in coders:
        arb_sheets[coder] = workbook.add_worksheet(f'Arbitration_{coder}')
        arb_sheets[coder].write_row(0, 0, arb_header)

    # Discussion

    # CSV Headers and Writers
    arb_writer = csv.writer(arbfile, delimiter=',')
    arb_writer.writerow(arb_header)

    writer = csv.DictWriter(codefile, code_header, delimiter=',')
    writer.writeheader()

    individual_writer = csv.DictWriter(indivfile, individual_header, delimiter=',')
    individual_writer.writeheader()

    confusion_writer = csv.writer(confusionfile, delimiter=',')

    return stats_sheet, confusion_sheet, arb_sheets, individual_sheet, \
        writer, confusion_writer, arb_writer, individual_writer, code_header, \
        individual_header


def process_aggregate_codesheet(data_dict, analysis_codes, lead_code):

    # Get all codes from coders
    all_dfs = []
    for coder, data in data_dict.items():
        df = data[analysis_codes]
        all_dfs += [df]

    # Fill in that dataframe with the matching codes from coders
    coding_array = []
    sample_df = df
    for index in tqdm(range(sample_df.shape[0])):

        row_vals = []
        for df in all_dfs:
            row = df.iloc[index]
            if not pd.isnull(row[lead_code]):
                row_vals += row.values.tolist()
        row_vals += (len(all_dfs) * len(analysis_codes) - len(row_vals)) * [np.nan]
        coding_array += [row_vals]

    colnames = []
    for i in range(len(all_dfs)):
        colnames += [f'{code}_{i}' for code in analysis_codes]

    combined_df = pd.DataFrame(coding_array, columns=colnames)
    combined_df = remove_bad_rows(combined_df, '0', '1', lead_code)

    return combined_df


def process_pair_codesheet(data_dict, coders, analysis_codes, coder1, coder2, lead_code):

    dfs = []
    for coder, pair_coder in [[coder1, coder2], [coder2, coder1]]:
        df = data_dict[coder][coders + analysis_codes]
        df = df.astype({c: 'bool' for c in coders})
        df = df[(df[coder]) & (df[pair_coder])][analysis_codes]
        df = df.rename(columns={x: f'{x}_{coder}' for x in list(df)})
        dfs += [df]
        
    combined_df = pd.concat(dfs, axis=1)
    combined_df = remove_bad_rows(combined_df, coder1, coder2, lead_code)

    return combined_df


def remove_bad_rows(df, coder1, coder2, lead_code):

    """ Modify this to work for more than 2 coders with df.query
    """

    # Remove null rows that have not yet been coded, according to the "lead code" (i.e. first code).
    df = df[~(df[f'{lead_code}_{coder1}'].isna()) & ~(df[f'{lead_code}_{coder2}'].isna())]

    return df


def calculate_all_scores(
        output_row, df, coder1, coder2, code_dict, code, code_hierarchy,
        multi_select_codes, code_groups, lead_code, exclude_values):

    col1 = f'{code}_{coder1}'
    col2 = f'{code}_{coder2}'

    # Remove 'Unclear' and other excluded rows.
    df = df[~(df[col1].isin(exclude_values)) & ~(df[col2].isin(exclude_values))]

    # Basic Agreement
    output_row = calculate_agreement_scores(
        output_row, df,
        col1, col2, code_dict, code, prefix='')

    # Partial Agreement
    if code in multi_select_codes:
        if code in code_hierarchy:
            # Not generalizable to other data, obviously
            condition_code = code_hierarchy[code]
            h_data = df.copy()
            for key, item in condition_code.items():
                h_data = h_data[(h_data[f'{key}_{coder1}'] != item) & (h_data[f'{key}_{coder2}'] != item)]
            output_row = calculate_agreement_scores(
                output_row, h_data, col1,
                col2, code_dict, code, prefix='Partial_', partial=True)
        else:
            output_row = calculate_agreement_scores(
                output_row, df, col1, col2,
                code_dict, code, prefix='Partial_', partial=True)

    # Conditional Agreement
    if code in code_hierarchy:
        output_row = calculate_agreement_scores(
            output_row, h_data,
            col1, col2, code_dict, code, prefix='Conditional_')
    else:
        output_row['Conditional_Agreement'] = None
        output_row['Conditional_Cohen_Kappa'] = None

    # Grouped Agreement
    if code in code_groups:
        group_dict = code_groups[code]
        for key, item in group_dict.items():
            df = df.replace(key, item)
        grouped_categories = list(set(group_dict.values()))
        output_categories = grouped_categories + [x for x in code_dict[code] if x not in group_dict]
        output_dict = {code: list(output_categories)}
        output_row = calculate_agreement_scores(
            output_row, df,
            col1, col2, output_dict, code, prefix='Grouped_')
    else:
        output_row['Grouped_Agreement'] = None
        output_row['Grouped_Cohen_Kappa'] = None

    return output_row


def calculate_agreement_scores(
        output_row, df, col1, col2,
        code_dict, code, prefix='', partial=False):

    # Total Rows
    count = df.shape[0]
    output_row[f'{prefix}N'] = count

    if partial:
        # This is so messed up. Means to an end.
        data1 = []
        data2 = []
        same_count = 0
        for _, row in df.iterrows():
            vals1, vals2 = str.split(str(row[col1]), '|'), str.split(str(row[col2]), '|')
            if not set(vals1).isdisjoint(vals2):
                vals1, vals2 = vals1[0], vals1[0]
                same_count += 1
            else:
                vals1, vals2 = vals1[0], vals2[0]
            data1 += [vals1]
            data2 += [vals2]
    else:
        data1 = df[col1].astype(str)
        data2 = df[col2].astype(str)
        same_count = df[df[col1] == df[col2]].shape[0]

    # Agreement
    agreement = same_count / count
    output_row[f'{prefix}Agreement'] = agreement

    # Cohen's Kappa
    c_kappa = cohen_kappa_score(data1, data2, labels=code_dict[code])
    if np.isnan(c_kappa):
        output_row[f'{prefix}Cohen_Kappa'] = 'N/A'
    else:
        output_row[f'{prefix}Cohen_Kappa'] = c_kappa

    # Fleiss' Kappa

    return output_row


def write_confusion(
        df, writer, confusion_sheet, confusion_rownum,
        code, coder1, coder2, code_dict, multi_select_codes):

    """ Writes confusion matrices into something tractable in a .csv file.
        Enable for 2+ raters
    """

    col1 = list(df[f'{code}_{coder1}'].astype(str))
    col2 = list(df[f'{code}_{coder2}'].astype(str))

    # # Remove 'Unclear' and other excluded rows.
    # df = df[~(df[col1].isin(exclude_values)) & ~(df[col2].isin(exclude_values))]

    # Deal with multi-select in the confusion matrices. Not sure if redundant
    if code in multi_select_codes:
        new_cols = [[], []]
        for idx, value1 in enumerate(col1):
            value2 = col2[idx]
            if '|' in value1 or '|' in value2:
                value1 = set(str.split(value1, '|'))
                value2 = set(str.split(value2, '|'))
                diff_vals = (value1 - value2).union(value2 - value1)
                match_vals = value1.intersection(value2)
                for match in match_vals:
                    new_cols[0] += [match]
                    new_cols[1] += [match]
                for diff in (value1 - value2):
                    for val in value2:
                        new_cols[0] += [diff]
                        new_cols[1] += [val]
                for diff in (value2 - value1):
                    for val in value1:
                        new_cols[0] += [val]
                        new_cols[1] += [diff]
        col1 += new_cols[0]
        col2 += new_cols[1]

    # Confusion matrix
    confusion = confusion_matrix(col1, col2, labels=code_dict[code])

    # No ground truth, so make it symmetric
    # 20 points if you can find a Python-y way to do this.
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            if i != j:
                total = confusion[i, j] + confusion[j, i]
                confusion[i, j] = total
                confusion[j, i] = total
            
    writer.writerow([code])
    labels = code_dict[code]

    code_row = [code]
    writer.writerow(code_row)
    confusion_rownum = write_xlsx_row(confusion_sheet, code_row, confusion_rownum)

    header_row = [''] + labels
    writer.writerow(header_row)
    confusion_rownum = write_xlsx_row(confusion_sheet, header_row, confusion_rownum)

    for idx, row in enumerate(confusion):
        output_row = [labels[idx]] + row.tolist()
        confusion_sheet.write_row(confusion_rownum, 0, output_row)
        confusion_rownum += 1
        writer.writerow(output_row)

    return confusion_rownum


def write_arb(
        writer, data_dict, lead_code, arb_cols,
        codes_only, arb_sheets, arb_rownums):

    """ Identify codes without a majority consensus, and write those to a new sheet for arbitrating.
        Currently exits arb when any arbitrator affirms the choice of a previous coder, so
        more complex majority systems are not implemented.
    """

    sample_df = data_dict[list(data_dict.keys())[0]]

    for i in tqdm(range(sample_df.shape[0])):

        # See who has and has not coded this data.
        have_coded = []
        arbitrators = []
        for coder, data in data_dict.items():
            row = data.iloc[i]
            if not pd.isnull(row[lead_code]):
                have_coded += [coder]
            else:
                arbitrators += [coder]

        # If 2+ people have coded...
        if len(have_coded) > 1:

            # Grab the relevant data...
            if arbitrators:
                arbitrator = random.choice(arbitrators)
            else:
                arbitrator = 'Discussion'

            output_rows = []
            code_rows = []
            for coder in have_coded:
                output_row = data_dict[coder].iloc[i].fillna('')
                data_cols = output_row[arb_cols].values.tolist()
                code_cols = output_row[codes_only].values.tolist()
                code_rows += [code_cols]
                output_rows += [data_cols + [coder, 'FALSE'] + code_cols]

            # Wonky method to determine if there is a majority.
            # There's probably a better way..
            # Hack here for the Notes column, TODO
            # majority_dict = defaultdict(int)
            majority = False
            for idx, row in enumerate(code_rows):
                for row2 in code_rows[idx + 1:]:
                    if row[:-1] == row2[:-1]:
                        majority = True

            # Finally, if there is no majority, write to arb file
            # if any([value > 0 for key, value in majority_dict.items()]):
            if majority:
                continue

            for output_row in output_rows:
                writer.writerow(output_row)
                arb_rownums[arbitrator] = write_xlsx_row(
                    arb_sheets[arbitrator],
                    output_row, arb_rownums[arbitrator])

            arb_row = [''] * len(arb_cols) + [arbitrator, 'TRUE']

            # Blank out disagreements for arbitrator
            # Very confusing. I'm a little out of it right now tbh.
            for idx in range(len(code_cols)):
                answers = [row[idx] for row in code_rows]
                if len(answers) == len(set(answers)):
                    arb_row += ['']
                else:
                    arb_row += [max(set(answers), key=answers.count)]

            writer.writerow(arb_row)
            arb_rownums[arbitrator] = write_xlsx_row(
                arb_sheets[arbitrator],
                arb_row, arb_rownums[arbitrator])

    return arb_sheets, arb_rownums


def process_individual_stats(
        individual_writer, individual_sheet, individual_rownum,
        data_dict, lead_code, analysis_codes, multi_select_codes,
        individual_header):

    for coder, df in data_dict.items():

        output_row = {'Coder': coder}

        df = df[~df[lead_code].isnull()]

        for code in analysis_codes:

            output_row['Code'] = code

            sub_df = df[code].dropna()

            # Deal with multi-select in the confusion matrices. Not sure if redundant
            if code in multi_select_codes:
                data = list(sub_df.astype(str))
                multi_select_count = 0
                new_data = []
                for idx, value in enumerate(data):
                    if '|' in value:
                        multi_select_count += 1
                        values = set(str.split(value, '|'))
                        for val in values:
                            new_data += [val]
                    else:
                        new_data += [value]
                sub_df = pd.DataFrame(new_data, columns=[code])
                sub_df = sub_df[code]
                multi_select_percent = multi_select_count / len(data)
            else:
                multi_select_count = None
                multi_select_percent = None

            output_row['Multi_N'] = multi_select_count
            output_row['Multi_Percent'] = multi_select_percent

            # sub_df = sub_df[~sub_df[code].isnull()]
            percents = sub_df.value_counts(normalize=True) * 100
            counts = sub_df.value_counts()

            for key, value in percents.items():
                output_row['N'] = counts[key]
                output_row['Percent'] = value
                output_row['Value'] = key

                individual_writer.writerow(output_row)
                write_xlsx_row(individual_sheet, output_row, individual_rownum, individual_header)
                individual_rownum += 1

    return


def write_discussion(dworkbook, data_dict, lead_code, codes_only, coders, analysis_codes):

    # Individual Statistics
    discussion_header = ['Pair_1', 'Pair_2', 'Row'] + analysis_codes

    with open('Test_Count.csv', 'a') as f:
        writer = csv.writer(f, delimiter=',')

        for idx, (coder1, coder2) in enumerate(combinations(coders, 2)):

            discussion_sheet = dworkbook.add_worksheet(f'{coder1}_{coder2}')
            discussion_sheet.write_row(0, 0, discussion_header)

            combined = process_pair_codesheet(
                data_dict, coders, analysis_codes, coder1, coder2, lead_code)

            rownum = 1
            for i in tqdm(range(combined.shape[0])): 

                row = combined.iloc[i]
                output_row = {'Pair_1': coder1, 'Pair_2': coder2, 'Row': combined.index[i]}

                if row.iloc[0:len(analysis_codes)].tolist() == row.iloc[len(analysis_codes):].tolist():
                    writer.writerow([combined.index[i], 0])
                    continue

                wrong_codes = 0
                for code in analysis_codes:
                    if row[f'{code}_{coder1}'] != row[f'{code}_{coder2}']:
                        output_row[code] = ' // '.join([str(row[f'{code}_{coder1}']), str(row[f'{code}_{coder2}'])])
                        wrong_codes += 1
                writer.writerow([combined.index[i], wrong_codes])
                
                write_xlsx_row(discussion_sheet, output_row, rownum, discussion_header)
                rownum += 1

    return


def scratch_code():

            # params = namedtuple('Parameters', [
                # 'code_dict', 'code_hierarchy', 'multi_select_codes', 'code_groups',
                # 'lead_code', 'exclude_values', 'analysis_codes', 'code_header', 'max_raters'])

    return


if __name__ == '__main__':
    pass
