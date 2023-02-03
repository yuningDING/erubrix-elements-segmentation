import argparse
import os

import pandas as pd


###
# Get all the E-mails as txt data. E-mails without content will not be saved.
# Input: ../e-rubrix_fixed.xlsx
# Output: ./data/all with <E-mail-ID>.txt
###


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", type=str, required=True)
    return parser.parse_args()


def write_files(prompt, ids, df):
    count_file = 0
    path = '../data/' + str(prompt) + '/'
    if os.path.exists(path) is False:
        os.makedirs(path)
    for i in ids:
        text = df.loc[df['Rand_id'] == i, 'Text_complete'].iloc[0]
        if text.strip() == '[no text]':
            print(i)
            continue
        with open(path + i + '.txt', 'w', encoding='utf-8') as f:
            f.write(text)
            count_file += 1
    return count_file


print('Loading E-mails ....')
args = parse_args()
df_xlsx = pd.read_excel(args.xlsx)
count = write_files('../data/all', df_xlsx['Rand_id'].unique(), df_xlsx)
print(str(count) + ' E-mails are loaded!')
