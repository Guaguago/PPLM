import csv
import numpy as np
# import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

MAX_FILES = 4


def get_data(filename):
    csvfile = open(filename)
    reader = csv.reader(csvfile)

    data = []
    for i, row in enumerate(reader):
        if i == 0:
            headers = row
        else:
            data.append(row)
    csvfile.close()
    return headers, data


def decode(st):
    ints = [int(s) for s in st.split('_')]
    # Version 2
    ii, j1, j2 = ints[0], np.mod(ints[1], MAX_FILES), np.mod(ints[2], MAX_FILES)
    return ii, j1, j2


# p-value of two binomial distributions
# one sided tail
def two_samp(x1, x2, n1, n2):
    p1 = x1 / n1
    p2 = x2 / n2
    phat = (x1 + x2) / (n1 + n2)
    z = (p1 - p2) / np.sqrt(phat * (1 - phat) * (1 / n1 + 1 / n2))
    return stats.norm.sf(np.abs(z))


def print_info_t(scores, counts):
    percs = scores / counts
    print('total counts, on topic counts, percentages:')
    for i in range(MAX_FILES):
        print('{},{},{}'.format(counts[i], scores[i], percs[i]))

    pvalues = np.zeros((MAX_FILES, MAX_FILES))
    for i in range(MAX_FILES):
        for j in range(i, MAX_FILES):
            dist_i = [1] * scores[i] + [0] * (counts[i] - scores[i])
            dist_j = [1] * scores[j] + [0] * (counts[j] - scores[j])
            pvalue = two_samp(scores[i], scores[j], counts[i], counts[j])
            pvalues[i, j] = pvalue
            pvalues[j, i] = pvalue
    for row in pvalues:
        print('{:.8f},{:.8f},{:.8f},{:.8f}'.format(row[0], row[1], row[2], row[3]))


def get_counts_indices(data, order_index, label_indices):
    scores = np.zeros(MAX_FILES, dtype=int)
    counts = np.zeros(MAX_FILES, dtype=int)
    skipped = 0
    for rownum, row in enumerate(data):
        order = row[order_index]
        for label_index in label_indices:
            label = row[label_index].lower()
            if len(order) > 0 and len(label) > 0:
                a_cat, b_cat = decode(order)[1:]
                # print(label, order, a_cat, b_cat)
                if label == 'a' or label == 'both':
                    scores[a_cat] += 1
                if label == 'b' or label == 'both':
                    scores[b_cat] += 1
                counts[a_cat] += 1
                counts[b_cat] += 1
                if label not in ['a', 'b', 'both', 'neither']:
                    print('******invalid label: {}'.format(label))
            else:
                # print('empty label; skipping', rownum)
                skipped += 1
    print('skipped {}'.format(skipped))
    print_info_t(scores, counts)
    return scores, counts


# vote by row. each row contributes to one count (and 0 or 1 score based on majority vote)
def get_counts_vote_row(data, order_index, label_indices):
    scores = np.zeros(MAX_FILES, dtype=int)
    counts = np.zeros(MAX_FILES, dtype=int)
    skipped = 0
    for rownum, row in enumerate(data):
        order = row[order_index]
        if len(order) == 0:
            skipped += 1
        else:
            a_cat, b_cat = decode(order)[1:]
            row_score_a, row_score_b, row_counts = 0, 0, 0
            for label_index in label_indices:
                label = row[label_index].lower()
                if len(label) > 0:
                    if label == 'a' or label == 'both':
                        row_score_a += 1
                    if label == 'b' or label == 'both':
                        row_score_b += 1
                    row_counts += 1
                    if label not in ['a', 'b', 'both', 'neither']:
                        print('******invalid label: {}'.format(label))
                else:
                    print('empty label for nonempty prompt', rownum)
            # update big points
            if row_counts == 3:
                scores[a_cat] += row_score_a // 2
                scores[b_cat] += row_score_b // 2
                counts[a_cat] += 1
                counts[b_cat] += 1
            else:
                print('incomplete row...')
    print('skipped {}'.format(skipped))
    print_info_t(scores, counts)
    return scores, counts


# vote by sample. each sample contributes to one count (and 0 or 1 score based on majority vote)
# each sample should appear 9 times
def get_counts_vote_all(data, order_index, label_indices):
    samples = defaultdict(list)  # key = (sample string, category), value = votes (list of 9)
    skipped = 0
    for rownum, row in enumerate(data):
        order = row[order_index]
        if len(order) == 0:
            skipped += 1
        else:
            a_cat, b_cat = decode(order)[1:]
            sample_a, sample_b = row[0], row[1]
            for label_index in label_indices:
                label = row[label_index].lower()
                if len(label) > 0:
                    if label == 'a':
                        samples[(sample_a, a_cat)].append(1)
                        samples[(sample_b, b_cat)].append(0)
                    elif label == 'b':
                        samples[(sample_a, a_cat)].append(0)
                        samples[(sample_b, b_cat)].append(1)
                    elif label == 'both':
                        samples[(sample_a, a_cat)].append(1)
                        samples[(sample_b, b_cat)].append(1)
                    elif label == 'neither':
                        samples[(sample_a, a_cat)].append(0)
                        samples[(sample_b, b_cat)].append(0)
                    else:
                        print('******invalid label: {}'.format(label))
                else:
                    print('empty label for nonempty prompt', rownum)
    print('skipped {}'.format(skipped))
    dist = np.zeros((MAX_FILES, 10), dtype=int)
    for sample in samples:
        cat = sample[1]
        samp_scores = samples[sample]
        if len(samp_scores) != 9:
            print('something had {} votes'.format(len(samp_scores)))
            print(sample)
        dist[cat, np.array(samp_scores).sum()] += 1
    print(dist)
    scores = np.zeros(MAX_FILES, dtype=int)
    counts = np.zeros(MAX_FILES, dtype=int)
    # print_info_t(scores, counts)
    return scores, counts


def print_info_f_lists(scorelist):
    print('mean, stdev, min, max, counts:')
    for i in range(MAX_FILES):
        print('{},{},{},{},{}'.format(np.mean(scorelist[i]), np.std(scorelist[i]),
                                      np.min(scorelist[i]), np.max(scorelist[i]), len(scorelist[i])))

    pvalues = np.zeros((MAX_FILES, MAX_FILES))
    for i in range(MAX_FILES):
        for j in range(i, MAX_FILES):
            pvalue = stats.ttest_ind(scorelist[i], scorelist[j]).pvalue
            pvalues[i, j] = pvalue
            pvalues[j, i] = pvalue
    print('p-values')
    for row in pvalues:
        print('{:.8f},{:.8f},{:.8f},{:.8f}'.format(row[0], row[1], row[2], row[3]))


def get_fluencies_indices(data, order_index, label_indices, topic, dst_dir):
    scorelist = [[], [], [], []]
    variants = [set(), set(), set(), set()]
    skipped = 0
    for r, row in enumerate(data):
        order = row[order_index]
        if len(order) == 0:
            continue
        for label_ind_pair in label_indices:
            a_cat, b_cat = decode(order)[1:]
            cats = decode(order)[1:]
            group(variants, row, a_cat, b_cat)
            for i, ind in enumerate(label_ind_pair):
                label = row[ind]
                if len(label) > 0:
                    scorelist[cats[i]].append(int(label))
                else:
                    skipped += 1
    write_to_files(variants, topic, dst_dir)
    print('skipped {}'.format(skipped))
    print_info_f_lists(scorelist)
    return scorelist


def group(variants, row, a_cat, b_cat):
    variants[a_cat].add(row[0])
    variants[b_cat].add(row[1])


def seperate_samples(topic, src_dir, dst_dir):
    # hardcoded indices
    category_index = -1  # index of encoded seed and methods
    topic_indices = [2, 6, 10]
    fluency_indices = [(3, 4), (7, 8), (11, 12)]

    all_scores = np.zeros(MAX_FILES, dtype=int)
    all_counts = np.zeros(MAX_FILES, dtype=int)
    percs_ordered = np.zeros((len(file_info), MAX_FILES))  # percents saved in same order as file names
    for i, fname in enumerate(file_info):
        filename = src_dir + fname
        headers, data = get_data(filename)
        print(fname)
        scores, counts = get_counts_vote_row(data, category_index, topic_indices)
        #     scores, counts = get_counts_vote_all(data, category_index, topic_indices) # voting out of 9
        all_scores += scores
        all_counts += counts
        percs_ordered[i] = 100 * scores / counts
        print()
    print('all:')
    print_info_t(all_scores, all_counts)
    print('\n------------\n')

    # uber labeled fluencies
    all_fluencies = [[], [], [], []]
    for fname in file_info:
        filename = src_dir + fname
        headers, data = get_data(filename)
        print(fname)
        new_scores = get_fluencies_indices(data, category_index, fluency_indices, topic, dst_dir)
        for i in range(len(all_fluencies)):
            all_fluencies[i].extend(new_scores[i])
        print()
    print('all:')
    print_info_f_lists(all_fluencies)
    print('total counts')
    for x in all_fluencies:
        print(len(x))


def write_to_files(variants, topic, dst_dir):
    # method_labels = ['baseline (B)', 'gradient (BC)', 'baseline+reranking (BR)', 'gradient+reranking (BCR)']
    method_labels = ['B', 'BC', 'BR', 'BCR']
    ppl_scores, dist_scores = [], []
    for i, variant in enumerate(variants):
        if topic == 'BC' or 'BC':
            label = method_labels[i]
            with open('../../automated_evaluation/{}/{}'.format(topic, label), 'w') as f:
                [f.write('<|endoftext|>' + s) for s in variant]

        # from test.utils.ppl_scores import score_py
        # from test.utils.dist_scores import eval_distinct
        # import statistics as stat
        # # ppl_scores.append(stat.mean([score_trans(s) for s in variants[i]]))
        # ppl_scores.append(stat.mean([score_py(s[13:]) for s in variants[i]]))
        # dist_scores.append(eval_distinct(variants[i], 1))

    # print(topic)
    # print('ppl, dist-1')
    # print('  B:{} {}'.format(ppl_scores[0], dist_scores[0]))
    # print(' BR:{} {}'.format(ppl_scores[2], dist_scores[2]))
    # print(' BC:{} {}'.format(ppl_scores[1], dist_scores[1]))
    # print('BCR:{} {}'.format(ppl_scores[3], dist_scores[3]))

    # for j, item in enumerate(variants[i]):
    #     from copy_ppl import score_py
    #     import statistics as stat
    #     score_py(item[13:])

    # with open('{}/{}/'.format(dst_dir, topic) + method_labels[i], 'a') as f:
    #     # str = item.strip()[13:].replace('\n', '')
    #     # str = item.strip()[13:]
    #     f.write("%s" % item)


if __name__ == '__main__':
    src_dir = '../../human_annotation/pplm_labeled_csvs/'
    dst_dir = '../../automated_evaluation/'
    file_info = [
        # 'computers.csv',
        # 'legal.csv',
        # 'military.csv',
        # 'politics.csv',
        # 'religion.csv',
        # 'science.csv',
        # 'space.csv',
        # 'BC.csv',
        'BC.csv',
        # 'clickbait.csv'
    ]
    topic = file_info[0][:-4]
    seperate_samples(topic, src_dir, dst_dir)
