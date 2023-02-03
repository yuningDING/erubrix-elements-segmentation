import os
import random
import pandas as pd
from sklearn.model_selection import KFold


def get_k_folds(k, random_state, id_set):
    k_folds_train = []
    k_folds_test = []
    kf = KFold(n_splits=k, random_state=random_state, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(id_set)):
        # print(f"Fold {i}:")
        # print(f"  Amount Train: {len(train_index)}")
        # print(f"  Amount Test:  {len(test_index)}")
        k_folds_train.append([id_set[x] for x in train_index])
        k_folds_test.append([id_set[x] for x in test_index])
    return k_folds_train, k_folds_test


def get_gold(save_dir, fild_name, id_list, gold):
    if os.path.isdir(save_dir) is False:
        os.makedirs(save_dir)
    file_gold = gold[gold['file_name'].isin(id_list)]
    file_gold.to_csv(save_dir + '/' + fild_name, index=False)


RANDOM_SEED = 42
random.seed(RANDOM_SEED)

print('Splitting data sets...')
all_gold = pd.read_csv('../data/all/gold.csv')
print('Total amount of E-Mails:', len(all_gold['file_name'].unique()))

prompt_1 = all_gold[all_gold['file_name'].str.contains('A1')]
print('Prompt 1:', len(prompt_1['file_name'].unique()))
prompt_2 = all_gold[all_gold['file_name'].str.contains('A2')]
print('Prompt 2:', len(prompt_2['file_name'].unique()))
prompt_3 = all_gold[all_gold['file_name'].str.contains('A3')]
print('Prompt 3:', len(prompt_3['file_name'].unique()))


print('---[Prompt Specific]---')
dir = '../data/setting_prompt_specific/'
dir_p1 = dir + 'prompt_1/'
dir_p2 = dir + 'prompt_2/'
dir_p3 = dir + 'prompt_3/'


print('Split Validation')
print('-Prompt 1-')
prompt_1_validate = random.sample(list(prompt_1['file_name'].unique()),int(len(prompt_1['file_name'].unique()) / 10))
get_gold(dir_p1, 'validate.csv', list(prompt_1_validate), all_gold)
print(f"  Amount Validation: {len(prompt_1_validate)}")
print('-Prompt 2-')
prompt_2_validate = random.sample(list(prompt_2['file_name'].unique()),int(len(prompt_2['file_name'].unique()) / 10))
get_gold(dir_p2, 'validate.csv', list(prompt_2_validate), all_gold)
print(f"  Amount Validation: {len(prompt_2_validate)}")
print('-Prompt 3-')
prompt_3_validate = random.sample(list(prompt_3['file_name'].unique()),int(len(prompt_3['file_name'].unique()) / 10))
get_gold(dir_p3, 'validate.csv', list(prompt_3_validate), all_gold)
print(f"  Amount Validation: {len(prompt_3_validate)}")

prompt_1_without_validation = [item for item in list(prompt_1['file_name'].unique()) if item not in prompt_1_validate]
prompt_2_without_validation = [item for item in list(prompt_2['file_name'].unique()) if item not in prompt_2_validate]
prompt_3_without_validation = [item for item in list(prompt_3['file_name'].unique()) if item not in prompt_3_validate]

prompt_1_train, prompt_1_test = get_k_folds(10, RANDOM_SEED, prompt_1_without_validation)
prompt_2_train, prompt_2_test = get_k_folds(10, RANDOM_SEED, prompt_2_without_validation)
prompt_3_train, prompt_3_test = get_k_folds(10, RANDOM_SEED, prompt_3_without_validation)

for i in range(10):
    print(f"Fold {i}:")
    print('-Prompt 1-')
    get_gold(dir_p1 + 'fold_' + str(i), 'train.csv', list(prompt_1_train[i]), all_gold)
    print(f"  Amount Train: {len(list(prompt_1_train[i]))}")
    get_gold(dir_p1 + 'fold_' + str(i), 'test.csv', list(prompt_1_test[i]), all_gold)
    print(f"  Amount Test: {len(list(prompt_1_test[i]))}")
    print('-Prompt 2-')
    get_gold(dir_p2 + 'fold_' + str(i), 'train.csv', list(prompt_2_train[i]), all_gold)
    print(f"  Amount Train: {len(list(prompt_2_train[i]))}")
    get_gold(dir_p2 + 'fold_' + str(i), 'test.csv', list(prompt_2_test[i]), all_gold)
    print(f"  Amount Test: {len(list(prompt_2_test[i]))}")
    print('-Prompt 3-')
    get_gold(dir_p3 + 'fold_' + str(i), 'train.csv', list(prompt_3_train[i]), all_gold)
    print(f"  Amount Train: {len(list(prompt_3_train[i]))}")
    get_gold(dir_p3 + 'fold_' + str(i), 'test.csv', list(prompt_3_test[i]), all_gold)
    print(f"  Amount Test: {len(list(prompt_3_test[i]))}")

print('---[Cross Prompt]---')
dir = '../data/setting_cross_prompt/'

print('-Train on Prompt 1-')
dir_p1 = dir + 'train_prompt_1/'
get_gold(dir_p1, 'train.csv', list(prompt_1_train[0]), all_gold)
print(f"  Amount Train: {len(list(prompt_1_train[0]))}")
prompt_1_validate_plus_test = [item for item in list(prompt_1['file_name'].unique()) if item not in list(prompt_1_train[0])]
get_gold(dir_p1, 'validate.csv', prompt_1_validate_plus_test, all_gold)
print(f"  Amount Validation: {len(prompt_1_validate_plus_test)}")
get_gold(dir_p1, 'test_p2.csv', list(prompt_2['file_name'].unique()), all_gold)
print(f"  Amount Test P2: {len(list(prompt_2['file_name'].unique()))}")
get_gold(dir_p1, 'test_p3.csv', list(prompt_3['file_name'].unique()), all_gold)
print(f"  Amount Test P3: {len(list(prompt_3['file_name'].unique()))}")

print('-Train on Prompt 2-')
dir_p2 = dir + 'train_prompt_2/'
get_gold(dir_p2, 'train.csv', list(prompt_2_train[0]), all_gold)
print(f"  Amount Train: {len(list(prompt_2_train[0]))}")
prompt_2_validate_plus_test = [item for item in list(prompt_2['file_name'].unique()) if item not in list(prompt_2_train[0])]
get_gold(dir_p2, 'validate.csv', prompt_2_validate_plus_test, all_gold)
print(f"  Amount Validation: {len(prompt_2_validate_plus_test)}")
get_gold(dir_p2, 'test_p1.csv', list(prompt_1['file_name'].unique()), all_gold)
print(f"  Amount Test P1: {len(list(prompt_1['file_name'].unique()))}")
get_gold(dir_p2, 'test_p3.csv', list(prompt_3['file_name'].unique()), all_gold)
print(f"  Amount Test P3: {len(list(prompt_3['file_name'].unique()))}")

print('-Train on Prompt 3-')
dir_p3 = dir + 'train_prompt_3/'
get_gold(dir_p3, 'train.csv', list(prompt_3_train[0]), all_gold)
print(f"  Amount Train: {len(list(prompt_3_train[0]))}")
prompt_3_validate_plus_test = [item for item in list(prompt_3['file_name'].unique()) if item not in list(prompt_3_train[0])]
get_gold(dir_p2, 'validate.csv', prompt_3_validate_plus_test, all_gold)
print(f"  Amount Validation: {len(prompt_3_validate_plus_test)}")
get_gold(dir_p3, 'test_p1.csv', list(prompt_1['file_name'].unique()), all_gold)
print(f"  Amount Test P1: {len(list(prompt_1['file_name'].unique()))}")
get_gold(dir_p3, 'test_p2.csv', list(prompt_2['file_name'].unique()), all_gold)
print(f"  Amount Test P2: {len(list(prompt_2['file_name'].unique()))}")


print('---[All with Prompt Sampling]---')
dir = '../data/setting_all/'
for i in range(10):
    print(f"Fold {i}:")
    train = []
    train.extend(list(prompt_1_train[i]))
    train.extend(list(prompt_2_train[i]))
    train.extend(list(prompt_3_train[i]))
    print(f"  Amount Train: {len(train)}")
    get_gold(dir + 'fold_' + str(i), 'train.csv', train, all_gold)

    test = []
    test.extend(list(prompt_1_test[i]))
    test.extend(list(prompt_2_test[i]))
    test.extend(list(prompt_3_test[i]))
    print(f"  Amount Test: {len(test)}")
    get_gold(dir + 'fold_' + str(i), 'test.csv', test, all_gold)

validation = []
validation.append(prompt_1_validate)
validation.append(prompt_2_validate)
validation.append(prompt_3_validate)
print(f"  Amount Validation: {len(validation)}")
get_gold(dir,'validate.csv', validation, all_gold)


print('---[All-Reduced with Prompt Sampling]---')
dir = '../data/setting_all_reduced/'

print('Split Validation')
validate_reduced = []
prompt_1_validate_reduced = random.sample(prompt_1_validate,int(len(prompt_1_validate) / 3))
prompt_2_validate_reduced = random.sample(prompt_2_validate,int(len(prompt_2_validate) / 3))
prompt_3_validate_reduced = random.sample(prompt_3_validate,int(len(prompt_2_validate) / 3))
validate_reduced.extend(prompt_1_validate_reduced)
validate_reduced.extend(prompt_2_validate_reduced)
validate_reduced.extend(prompt_3_validate_reduced)
get_gold(dir, 'validate.csv', validate_reduced, all_gold)
print(f"  Amount Validation: {len(prompt_2_validate)}")


prompt_1_train, prompt_1_test = get_k_folds(10, RANDOM_SEED, random.sample(prompt_1_without_validation,
                                                                           int(len(
                                                                               prompt_1_without_validation) / 3)))
prompt_2_train, prompt_2_test = get_k_folds(10, RANDOM_SEED, random.sample(prompt_2_without_validation,
                                                                           int(len(
                                                                               prompt_2_without_validation) / 3)))
prompt_3_train, prompt_3_test = get_k_folds(10, RANDOM_SEED, random.sample(prompt_3_without_validation,
                                                                           int(len(
                                                                               prompt_3_without_validation) / 3)))
for i in range(10):
    print(f"Fold {i}:")
    train = []
    train.extend(list(prompt_1_train[i]))
    train.extend(list(prompt_2_train[i]))
    train.extend(list(prompt_3_train[i]))
    print(f"  Amount Train: {len(train)}")
    test = []
    test.extend(list(prompt_1_test[i]))
    test.extend(list(prompt_2_test[i]))
    test.extend(list(prompt_3_test[i]))
    print(f"  Amount Test: {len(test)}")
    get_gold(dir + 'fold_' + str(i), 'train.csv', train, all_gold)
    get_gold(dir + 'fold_' + str(i), 'test.csv', test, all_gold)

