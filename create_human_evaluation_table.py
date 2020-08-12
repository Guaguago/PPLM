import pandas as pd
import numpy as np

MAX_FILES = 3
np.random.seed(0)


def shuffle(df_1, df_2, cat_1, cat_2):
    df = pd.DataFrame()
    df[0] = df_1
    df[1] = df_2
    for index, row in df.iterrows():

        A = np.random.randint(10, 200) * MAX_FILES + cat_1
        B = np.random.randint(10, 200) * MAX_FILES + cat_2

        switch = np.random.randint(0, 2)
        if switch:
            temp = df.loc[index, 0]
            df.loc[index, 0] = df.loc[index, 1]
            df.loc[index, 1] = temp
            temp = A
            A = B
            B = temp

        cat = '{}_{}_{}'.format(index, A, B)
        df.loc[index, 2] = cat
    # df.rename({0: 'A', 1: 'B', 2: 'codes'}, axis=1, inplace=True)
    return df


def create_human_anotation_table(models, anotations):
    df_1 = models[0][1]
    df_2 = models[1][1]
    df_3 = models[2][1]
    cat_1 = 0
    cat_2 = 1
    cat_3 = 2
    first = shuffle(df_1, df_2, cat_1, cat_2)
    second = shuffle(df_1, df_3, cat_1, cat_3)
    third = shuffle(df_2, df_3, cat_2, cat_3)

    df = pd.concat([first, second, third])
    df.rename({0: 'A', 1: 'B', 2: 'codes'}, axis=1, inplace=True)

    header_list = ['A', 'B',
                   'Which passage is more negative?', 'How fluent is the passage of A?',
                   'How fluent is the passage of B?',
                   '{}'.format(anotations[0]),
                   'Which passage is more negative?', 'How fluent is the passage of A?',
                   'How fluent is the passage of B?',
                   '{}'.format(anotations[1]),
                   'Which passage is more negative?', 'How fluent is the passage of A?',
                   'How fluent is the passage of B?',
                   '{}'.format(anotations[2]),
                   '', '', '', '', '', '', '', '', '', 'codes']
    return df.reindex(header_list, axis=1)



if __name__ == '__main__':
    model_1_samples = pd.read_csv('/Users/xuchen/Desktop/人工测评/pos-vad-loss.csv', header=None)
    model_2_samples = pd.read_csv('/Users/xuchen/Desktop/人工测评/pos-vad-loss.csv', header=None)
    model_3_samples = pd.read_csv('/Users/xuchen/Desktop/人工测评/pos-vad-loss.csv', header=None)

    df = create_human_anotation_table(
        [model_1_samples, model_2_samples, model_3_samples],
        ['XuChen', 'LiTong', 'LiuSiLiang']
    )

    df.to_csv('/Users/xuchen/Desktop/人工测评/out/out.csv', index=False)

