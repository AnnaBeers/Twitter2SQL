""" Scripts for analyzing coded data stored in Google Sheets. 
    Currently mostly works for one application.
"""

import xlsxwriter
import os
import random
import pandas as pd
import numpy as np
import csv

from sklearn.metrics import cohen_kappa_score, confusion_matrix
from pprint import pprint
from tqdm import tqdm
from collections import Counter, defaultdict


def analyze_codes(input_data, 
            coders,
            output_directory,
            suffix='',
            multi_select_codes=None,
            lead_code=None,
            code_hierarchy=None,
            exclude_codes=['Notes'],
            exclude_values=['Unclear'],
            arbitration_columns=None,
            code_groups=None,
            max_raters=2,
            aggregate_statistics=True,
            confusion_matrices=True,
            pairwise_statistics=True,
            arbitration=True,
            verbose=True):

    """ Input xlsx is expected to have the following format:
        1. A 'Codebook' sheet with vertical columns of titled codes.
        2. Coding sheets per coder titled 'Tweets_{coder}'
        3. Any number of unrelated sheets.

        Each coding sheet is expected to have, in order:
        1. Any number of data columns
        2. TRUE/FALSE coder columns = to # of coders.
        3. Code columns
    """

    code_level_stats = os.path.join(output_directory, f'Code_Level_Stats{suffix}.csv')
    confusion_matrix_stats = os.path.join(output_directory, f'Confusion_Matrices{suffix}.csv')
    confusion_matrix_image_dir = os.path.join(output_directory, f'Confusion_Matrices{suffix}')
    arbitration_sheet = os.path.join(output_directory, f'Arbitration{suffix}.csv')
    output_xlsx = os.path.join(output_directory, f'Coding_Analysis{suffix}.xlsx')

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
                    data_dict[coder] = raw_xlsx.parse(name, keep_default_na=False, na_values=[''])

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
    codes_coders = []
    past_coders = False
    for colname in list(data_dict[coders[0]]):
        if colname in coders:
            past_coders = True
        if past_coders:
            if colname not in coders:
                codes_only += [colname]
            codes_coders += [colname]
    analysis_codes = [x for x in codes_only if x not in exclude_codes]

    if lead_code is None:
        lead_code = analysis_codes[0]

    if max_raters is None:
        max_raters = len(coders)

    """ 3. Define Metrics
    """

    code_header = ['Combined Pair', 'Pair 1', 'Pair 2', 'Code']

    extra_measures = ['', 'Conditional_', 'Partial_', 'Grouped_']
    for extra in extra_measures:
        code_header += [f'{extra}N', f'{extra}Agreement', f'{extra}Cohen_Kappa']


    """ 4. Write out data.
    """

    # First figure out XLSX writing...
    with xlsxwriter.Workbook(output_xlsx) as workbook:

        statistics_worksheet, confusion_worksheet, arbitration_worksheets, arbitration_header = format_xlsx(workbook, 
                code_header, arbitration_columns, coders + ['Discussion'], codes_coders, codes_only, data_dict)

        confusion_rownum, statistics_rownum = 0, 1
        arbitration_rownums = {coder: 1 for coder in coders}

        # Then figure out CSV writing...
        with open(code_level_stats, 'w') as codefile, open(confusion_matrix_stats, 'w') as confusionfile, \
                    open(arbitration_sheet, 'w') as arbfile:

            arb_writer = csv.writer(arbfile, delimiter=',')
            arb_writer.writerow(arbitration_header)

            writer = csv.DictWriter(codefile, code_header, delimiter=',')
            writer.writeheader()

            confusion_writer = csv.writer(confusionfile, delimiter=',')


            # Get scores for the aggregate data first..
            if aggregate_statistics:
                coder1, coder2 = '_0', '_1'

                combined = process_aggregate_codesheet(data_dict, codes_coders, analysis_codes, lead_code)
                for code in tqdm(analysis_codes):
                    output_row = {'Combined Pair': 'All', 'Code': code}
                    output_row, confusion = calculate_all_scores(output_row, combined, coder1, coder2, code_dict,
                            code, code_hierarchy, multi_select_codes, code_groups, lead_code, exclude_values)

                    if confusion_matrices:
                        confusion_rownum = write_confusion(confusion_writer, confusion_worksheet, confusion_rownum,
                                confusion, code, code_dict)

                    list_row = [output_row[x]  if x in output_row else '' for x in code_header]
                    statistics_rownum = write_xlsx_row(statistics_worksheet, list_row, statistics_rownum)
                    writer.writerow(output_row)

            # Calculate scores for each coding pair.
            if pairwise_statistics:
                for idx, coder1 in enumerate(coders):
                    for coder2 in coders[idx + 1:]:

                        df1 = process_pair_codesheet(data_dict, codes_coders, analysis_codes, coder1, coder2)
                        df2 = process_pair_codesheet(data_dict, codes_coders, analysis_codes, coder2, coder1)
                        combined = pd.concat([df1, df2], axis=1)

                        for code in tqdm(analysis_codes):
                            output_row = {'Combined Pair': f'{coder1}_{coder2}', 'Pair 1': coder1, 'Pair 2': coder2,
                                    'Code': code}
                            output_row['Code'] = code
                            output_row, confusion = calculate_all_scores(output_row, combined, f'_{coder1}',
                                    f'_{coder2}', code_dict, code, code_hierarchy, 
                                    multi_select_codes, code_groups, lead_code, exclude_values)

                            list_row = [output_row[x]  if x in output_row else '' for x in code_header]
                            statistics_rownum = write_xlsx_row(statistics_worksheet, list_row, statistics_rownum)
                            writer.writerow(output_row)

            # Then write out rows for arbitration
            if arbitration and True:
                write_arbitration(arb_writer, data_dict, lead_code, arbitration_columns, \
                        codes_only, arbitration_worksheets, arbitration_rownums)

    return


def write_xlsx_row(sheet, row, rownum):

    sheet.write_row(rownum, 0, row)
    rownum += 1

    return rownum


def format_xlsx(workbook, code_header, arbitration_columns, 
            coders, codes_coders, codes_only, data_dict):

    statistics_worksheet = workbook.add_worksheet('Agreement_Statistics')
    statistics_worksheet.write_row(0, 0, code_header)

    confusion_worksheet = workbook.add_worksheet('Confusion_Statistics')

    if arbitration_columns is None:
        arbitration_header = list(data_dict[coders[0]])
    else:
        arbitration_header = [x for x in list(data_dict[coders[0]]) if x in arbitration_columns]

    arbitration_header = arbitration_header + ['Coder', 'Arbitrate?'] + codes_only
    arbitration_worksheets = {}
    for coder in coders:
        arbitration_worksheets[coder] = workbook.add_worksheet(f'Arbitration_{coder}')
        arbitration_worksheets[coder].write_row(0, 0, arbitration_header)

    return statistics_worksheet, confusion_worksheet, arbitration_worksheets, arbitration_header


def write_arbitration(writer, data_dict, lead_code, arbitration_columns, 
            codes_only, arbitration_worksheets, arbitration_rownums):

    """ Identify codes without a majority consensus, and write those to a new sheet for arbitrating.
        Currently exits arbitration when any arbitrator affirms the choice of a previous coder, so
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
                data_cols = output_row[arbitration_columns].values.tolist()
                code_cols = output_row[codes_only].values.tolist()
                code_rows += [code_cols]
                output_rows += [data_cols + [coder, 'FALSE'] + code_cols]

            # Wonky method to determine if there is a majority.
            # There's probably a better way..
            # Hack here for the Notes column, TODO
            majority_dict = defaultdict(int)
            majority = False
            for idx, row in enumerate(code_rows):
                for row2 in code_rows[idx + 1:]:
                    if row[:-1] == row2[:-1]:
                        majority = True

            # Finally, if there is no majority, write to arbitration file
            # if any([value > 0 for key, value in majority_dict.items()]):
            if majority:
                continue
            else:
                for output_row in output_rows:
                    writer.writerow(output_row)
                    arbitration_rownums[arbitrator] = write_xlsx_row(arbitration_worksheets[arbitrator], 
                            output_row, arbitration_rownums[arbitrator])


                arb_row = [''] * len(arbitration_columns) + [arbitrator, 'TRUE']

                # Blank out disagreements for arbitrator
                # Very confusing. I'm a little out of it right now tbh.
                for idx in range(len(code_cols)):
                    answers = [row[idx] for row in code_rows]
                    if len(answers) == len(set(answers)):
                        arb_row += ['']
                    else:
                        arb_row += [max(set(answers), key=answers.count)]

                writer.writerow(arb_row)
                arbitration_rownums[arbitrator] = write_xlsx_row(arbitration_worksheets[arbitrator], 
                        arb_row, arbitration_rownums[arbitrator])


    return


def write_confusion(writer, confusion_worksheet, confusion_rownum,
            confusion, code, code_dict):

    """ Writes confusion matrices into something tractable in a .csv file.
    """

    writer.writerow([code])
    labels = code_dict[code]

    code_row = [code]
    writer.writerow(code_row)
    confusion_rownum = write_xlsx_row(confusion_worksheet, code_row, confusion_rownum)

    header_row = [''] + labels
    writer.writerow(header_row)
    confusion_rownum = write_xlsx_row(confusion_worksheet, header_row, confusion_rownum)

    for idx, row in enumerate(confusion):
        output_row = [labels[idx]] + row.tolist()
        confusion_worksheet.write_row(confusion_rownum, 0, output_row)
        confusion_rownum += 1
        writer.writerow(output_row)

    return confusion_rownum


def process_aggregate_codesheet(data_dict, codes_coders, analysis_codes, lead_code):

    # Get all codes from coders
    all_dfs = []
    for key, data in data_dict.items():
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

    return combined_df



def process_pair_codesheet(data_dict, codes_coders, analysis_codes, coder, pair_coder):

    df = data_dict[coder][codes_coders]
    df = df[(df[coder]) & (df[pair_coder])][analysis_codes]
    df.columns = [f'{x}_{coder}' for x in list(df)]
    return df


def calculate_all_scores(output_row, df, coder1, coder2, code_dict, code, code_hierarchy, 
            multi_select_codes, code_groups, lead_code, exclude_values):

    col1 = f'{code}{coder1}'
    col2 = f'{code}{coder2}'

    # Remove null rows that have not yet been coded, according to the "lead code" (i.e. first code).
    df = df[~(df[f'{lead_code}{coder1}'].isna()) & ~(df[f'{lead_code}{coder2}'].isna())]

    # Remove 'Unclear' and other excluded rows.
    df = df[~(df[col1].isin(exclude_values)) & ~(df[col2].isin(exclude_values))]

    # Basic Agreement
    output_row, count, agreement, c_kappa, confusion = calculate_agreement_scores(output_row, df, 
            col1, col2, code_dict, code, prefix='')

    # Partial Agreement
    if code in multi_select_codes:
        if code in code_hierarchy:
            # Not generalizable to other data, obviously
            condition_code = code_hierarchy[code]
            h_data = df.copy()
            for key, item in condition_code.items(): 
                h_data = h_data[(h_data[f'{key}{coder1}'] != item) & (h_data[f'{key}{coder2}'] != item)]
            output_row, p_count, p_agreement, p_kappa, p_confusion = calculate_agreement_scores(output_row, h_data, col1,
                    col2, code_dict, code, prefix='Partial_', partial=True)
        else:
            output_row, p_count, p_agreement, p_kappa, p_confusion = calculate_agreement_scores(output_row, df, col1, col2,
                    code_dict, code, prefix='Partial_', partial=True)

    # Conditional Agreement
    if code in code_hierarchy:                    
        output_row, h_count, h_agreement, h_kappa, h_confusion = calculate_agreement_scores(output_row, h_data,
                col1, col2, code_dict, code, prefix='Conditional_')
    else:
        output_row['Conditional_Agreement'] = None
        output_row['Conditional_Cohen_Kappa'] = None

    # Grouped Agreement
    if code in code_groups:
        group_dict = code_groups[code]
        for key, item in group_dict.items():
            df = df.replace(key, item)
        grouped_categories = list(set([val for val in group_dict.values()]))
        output_categories = grouped_categories + [x for x in code_dict[code] if x not in group_dict]
        output_dict = {code: list(output_categories)}
        output_row, g_count, g_agreement, g_kappa, g_confusion = calculate_agreement_scores(output_row, df, 
                col1, col2, output_dict, code, prefix='Grouped_')

    return output_row, confusion


def calculate_agreement_scores(output_row, df, col1, col2, code_dict, code, prefix='', partial=False):

    # Total Rows
    count = df.shape[0]
    output_row[f'{prefix}N'] = count

    if partial:
        # This is so messed up. Means to an end.
        data1 = []
        data2 = []
        same_count = 0
        for index, row in df.iterrows():
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

    # Confusion matrix
    confusion = confusion_matrix(data1, data2, labels=code_dict[code])

    return output_row, count, agreement, c_kappa, confusion


if __name__ == '__main__':
    pass