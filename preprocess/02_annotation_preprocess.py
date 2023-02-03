import argparse

import pandas as pd

pd.options.mode.chained_assignment = None


###
# Preprocess of annotations
# Input: ../2022-09-02_Masterset_original_bereinigt.csv, ../data/all with <E-mail-ID>.txt
# Output: ../data/all/gold.csv
###


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    return parser.parse_args()


print('Loading Annotations ....')
# 1. read and sort annotation file
args = parse_args()
df = pd.read_csv(args.csv, encoding='utf-8')
df.drop(columns=df.columns[0], axis=1, inplace=True)
df = df.drop_duplicates()
df['cum'] = df.groupby('start').cumcount()
df = df.sort_values(by=['file_name', 'start', 'cum'])
df = df.drop(['ID', 'TextID', 'WordID', 'Preview', 'preview_length', 'cum'], axis=1)
df.reset_index(drop=True, inplace=True)

# 2. remove annotations on missing elements
for i in range(len(df)):
    if (
            "subject" or "question" or "Salutation" or "Matter" or "Concluding" or "Information" or "Closing") and "missing" in str(
            df['tag'][i]):
        df['tag'][i] = "Missing"
    elif "No task question addressed" in str(df['tag'][i]):
        df['tag'][i] = "Missing"
    elif "Subject" in str(df['tag'][i]):
        df['tag'][i] = "Subject line"
    elif "question" in str(df['tag'][i]):
        df['tag'][i] = "Questions"
    elif "Salutation" in str(df['tag'][i]):
        df['tag'][i] = "Salutation"
    elif "Matter" in str(df['tag'][i]):
        df['tag'][i] = "Matter of concern"
    elif "Concluding" in str(df['tag'][i]):
        df['tag'][i] = "Concluding sentence"
    elif "Information" in str(df['tag'][i]):
        df['tag'][i] = "Information about writer"
    elif "Closing" in str(df['tag'][i]):
        df['tag'][i] = "Closing"
    elif "Missing" in str(df['tag'][i]):
        df['tag'][i] = "Missing"

df = df[df['tag'] != 'Missing']
df.reset_index(drop=True, inplace=True)

# 3. add annotations 'Unknown' to the text without annotations, correct possible double annotations
double_anno = []
for i in range(1, len(df)):
    if df['file_name'][i] == df['file_name'][i - 1]:
        if df['start'][i] < df['end'][i - 1]:
            df['start'][i] = df['end'][i - 1] + 1
            if df['start'][i] > df['end'][i]:
                double_anno.append(i)
                df['end'][i] = df['end'][i - 1] + 1
        if df['start'][i] - df['end'][i - 1] > 2:
            new_row = {'start': df['end'][i - 1] + 1, 'end': df['start'][i] - 1,
                       'file_name': df['file_name'][i - 1], 'tag': 'Unknown'}
            df = df.append(new_row, ignore_index=True)
df = df.drop(double_anno)
print(str(len(double_anno)) + ' discourses are double annotated therefore deleted.')
df['cum'] = df.groupby('start').cumcount()
df = df.sort_values(by=['file_name', 'start', 'cum'])
df.reset_index(drop=True, inplace=True)

df["discourse"] = ""
file_id = df['file_name']
email_length = {}
for i in range(len(df)):
    try:
        with open(f"../data/all/{file_id[i]}.txt", encoding='utf-8') as f:
            discourse = f.read()
            if df['start'][i] > 0:
                start = df['start'][i] - 1
                end = df['end'][i] - 1
            else:
                start = df['start'][i]
                end = df['end'][i]
            df["discourse"][i] = discourse[int(start):int(end)]
            email_length[file_id[i]] = len(discourse.split())
    except FileNotFoundError:
        continue

# 4. convert annotations from character level to token level
predictions = []
file_name = file_id[0]
begin = 0
for idx, row in df.iterrows():
    index_list = []
    if row['file_name'] is not file_name:
        begin = 0
    length = len(row['discourse'].split())
    predictions.append(list(range(begin, begin + length)))
    begin = begin + length
    file_name = row['file_name']
df['prediction'] = predictions

# 5. remove unknown values, add discourse_id and reformat prediction string
df = df[df['tag'] != 'Unknown']
df = df.drop(['start', 'end', 'cum'], axis=1)
df.reset_index(drop=True, inplace=True)
df['discourse_id'] = ''
df['prediction_string'] = ''
annotation_length = {}
for file_id, discourses in df.groupby('file_name'):
    i = 0
    for idx, row in discourses.iterrows():
        df.at[idx, 'discourse_id'] = row['file_name'] + '_' + str(i)
        df.at[idx, 'prediction_string'] = str(row['prediction'])[1:-1].replace(',', ' ')
        i += 1
        annotation_length[file_id] = row['prediction'][-1:]
cols = ['file_name', 'discourse_id', 'discourse', 'tag', 'prediction_string']
df = df[cols]

df.to_csv('../data/all/gold.csv', index=False)
print(str(len(df)) + ' annotations are loaded!')
