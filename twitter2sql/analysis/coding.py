import os
import pandas as pd
import numpy as numpy
import csv

from sklearn.metrics import cohen_kappa_score
from pprint import pprint


def analyze_codes(input_data, 
            coders,
            output_directory,
            suffix='',
            multi_select_codes=None,
            lead_code=None,
            code_hierarchy=None,
            exclude_codes=['Notes'],
            code_groups=None):

    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    code_level_stats = os.path.join(output_directory, f'Code_Level_Stats{suffix}.csv')

    file_type = os.path.splitext(input_data)[1]
    if file_type not in ['.xlsx']:
        raise NotImplementedError

    if file_type == '.xlsx':

        raw_xlsx = pd.ExcelFile(input_data)
        print(raw_xlsx.sheet_names)

        data_dict = {}
        for coder in coders:
            for name in raw_xlsx.sheet_names:
                if coder in name:
                    data_dict[coder] = raw_xlsx.parse(name, keep_default_na=False, na_values=[''])

        code_dict = {}
        codebook = raw_xlsx.parse('Codebook', keep_default_na=False, na_values=[''])
        for col in list(codebook):
            code_dict[col] = list(codebook[col].dropna().astype(str))

    # This is really dumb.
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
            
    def process_codesheet(coder, pair_coder):

        df = data_dict[coder][codes_coders]
        df = df[(df[coder]) & (df[pair_coder])][analysis_codes]
        df.columns = [f'{x}_{coder}' for x in list(df)]
        return df

    code_header = ['Combined Pair', 'Pair 1', 'Pair 2', 'Code', 'N', 'Agreement', 'Cohen_Kappa']

    extra_measures = ['Conditional', 'Partial', 'Grouped']
    for extra in extra_measures:
        code_header += [f'{extra}_N', f'{extra}_Agreement', f'{extra}_Cohen_Kappa']

    with open(code_level_stats, 'w') as codefile:
        writer = csv.DictWriter(codefile, code_header, delimiter=',')
        writer.writeheader()

        for code in analysis_codes:
            output_row = {'Code': code}

            for idx, coder1 in enumerate(coders):
                for coder2 in coders[idx + 1:]:
                    # print(code, coder1, coder2)
                    output_row['Combined Pair'] = f'{coder1}_{coder2}'
                    output_row['Pair 1'] = coder1
                    output_row['Pair 2'] = coder2

                    df1 = process_codesheet(coder1, coder2)
                    df2 = process_codesheet(coder2, coder1)

                    combined = pd.concat([df1, df2], axis=1)
                    combined = combined[~(combined[f'{lead_code}_{coder1}'].isna()) & ~(combined[f'{lead_code}_{coder2}'].isna())]

                    col1 = f'{code}_{coder1}'
                    col2 = f'{code}_{coder2}'
                    cols = [col1, col2]

                    pd.set_option('display.max_rows', 100)

                    # Basic
                    output_row, count, agreement, c_kappa = calculate_agreement_scores(output_row, combined, col1, col2, code_dict, code, prefix='')

                    # Partial Agreement
                    if code in multi_select_codes:
                        if code in code_hierarchy:
                            condition_code = code_hierarchy[code]
                            h_data = combined[(combined[f'{condition_code}_{coder1}'] != 'No Theory') & (combined[f'{condition_code}_{coder2}'] != 'No Theory')]
                            p_output_row, p_count, p_agreement, p_kappa = calculate_partial_agreement_scores(output_row, h_data, col1, col2, code_dict, code, prefix='Partial_')
                        else:
                            p_output_row, p_count, p_agreement, p_kappa = calculate_partial_agreement_scores(output_row, combined, col1, col2, code_dict, code, prefix='Partial_')

                    # Conditional Agreement
                    if code in code_hierarchy:
                        # Fix this to be generalizable                        
                        h_output_row, h_count, h_agreement, h_kappa = calculate_agreement_scores(output_row, h_data, col1, col2, code_dict, code, prefix='Conditional_')
                    else:
                        output_row['Conditional_Agreement'] = None
                        output_row['Conditional_Cohen_Kappa'] = None

                    if code in code_groups:
                        group_dict = code_groups[code]
                        for key, item in group_dict.items():
                            combined = combined.replace(key, item)
                        output_categories = set([val for val in group_dict.values()])
                        output_dict = {code: list(output_categories)}
                        g_output_row, g_count, g_agreement, g_kappa = calculate_agreement_scores(output_row, combined, col1, col2, output_dict, code, prefix='Grouped_')


                    # Grouped Agreement


                    # Grouped Agreement


                    # if c_kappa < .2 and count > 10:
                    #     print(code_dict[code])
                    #     print(agreement, c_kappa)
                    #     print(subdata)

                    pprint(output_row)
                    writer.writerow(output_row)

    return


def calculate_agreement_scores(output_row, df, col1, col2, code_dict, code, prefix=''):

    count = df.shape[0]
    output_row[f'{prefix}N'] = count

    # Basic Agreement
    same_count = df[df[col1] == df[col2]].shape[0]
    agreement = same_count / count
    output_row[f'{prefix}Agreement'] = agreement

    # Cohen's Kappa
    c_kappa = cohen_kappa_score(df[col1].astype(str), df[col2].astype(str), labels=code_dict[code])
    output_row[f'{prefix}Cohen_Kappa'] = c_kappa

    return output_row, count, agreement, c_kappa


def calculate_partial_agreement_scores(output_row, df, col1, col2, code_dict, code, prefix=''):

    count = df.shape[0]
    output_row[f'{prefix}N'] = count

    # This is so messed up.
    partial_col1 = []
    partial_col2 = []
    same_count = 0
    for index, row in df.iterrows():
        vals1, vals2 = str.split(str(row[col1]), '|'), str.split(str(row[col2]), '|')
        if not set(vals1).isdisjoint(vals2):
            vals1, vals2 = vals1[0], vals1[0]
            same_count += 1
        else:
            vals1, vals2 = vals1[0], vals2[0]
        partial_col1 += [vals1]
        partial_col2 += [vals2]

    # Basic Agreement
    agreement = same_count / count
    output_row[f'{prefix}Agreement'] = agreement

    # Cohen's Kappa
    c_kappa = cohen_kappa_score(partial_col1, partial_col2, labels=code_dict[code])
    output_row[f'{prefix}Cohen_Kappa'] = c_kappa

    return output_row, count, agreement, c_kappa


if __name__ == '__main__':
    pass