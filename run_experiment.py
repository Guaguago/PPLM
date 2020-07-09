import neptune
from automated_evaluation.cal_ppl import cal_ppl
from automated_evaluation.cal_dist import cal_dist
from automated_evaluation.cal_acc import cal_acc


def samples(src):
    with open(src, 'r') as f:
        return f.read().split('<|endoftext|>')[1:]


if __name__ == '__main__':
    # set hyper parameters
    prefixes = [
        # standard 15 prefixes
        'Once upon a time', 'The book', 'The chicken', 'The city', 'The country',
        'The horse', 'The lake', 'The last time', 'The movie', 'The painting',
        'The pizza', 'The potato', 'The president of the country', 'The road', 'The year is 1910.',
        # # extra 35 prefixes
        # 'The article', 'I would like to', 'We should', 'In the future', 'The cat',
        # 'The piano', 'The walls', 'The hotel', 'The good news', 'The building',
        # 'The owner', 'Our house', 'Do you like', 'Her hair', 'The spider man',
        # 'The computer', 'My phone', 'The TV', 'The bus', 'Long long ago',
        # 'My daughter', 'The ice cream', 'This recipe', 'Most of us', 'The game',
        # 'The music', 'The show', 'The dress', 'In the evening', 'The traffic',
        # 'We usually', 'My mother', 'My dad', 'The meeting', 'My wife',
    ]
    pos_src = 'data/test/generated_samples/paper/vad_pos(2_150_10)'
    neg_src = 'data/test/generated_samples/paper/vad_neg(2_150_10)'
    method = 'BC_VAD'
    length = 50
    num_samples = 10
    num_iterations = 10
    total_samples = len(prefixes) * num_samples

    # obtain samples
    pos_samples = samples(pos_src)
    neg_samples = samples(neg_src)
    assert len(pos_samples) == total_samples
    assert len(neg_samples) == total_samples

    # calculate PPL
    pos_ppl = cal_ppl(pos_samples)
    neg_ppl = cal_ppl(neg_samples)
    mean_ppl = (pos_ppl + neg_ppl) / 2

    # calculate dist-1,2,3
    pos_dist_1 = cal_dist(pos_samples, 1)
    pos_dist_2 = cal_dist(pos_samples, 2)
    pos_dist_3 = cal_dist(pos_samples, 3)
    neg_dist_1 = cal_dist(neg_samples, 1)
    neg_dist_2 = cal_dist(neg_samples, 2)
    neg_dist_3 = cal_dist(neg_samples, 3)
    mean_dist_1 = (pos_dist_1 + neg_dist_1) / 2
    mean_dist_2 = (pos_dist_2 + neg_dist_2) / 2
    mean_dist_3 = (pos_dist_3 + neg_dist_3) / 2

    # calculate acc
    pos_acc = cal_acc(pos_samples, True)
    neg_acc = cal_acc(neg_samples, False)
    mean_acc = (pos_acc + neg_acc) / 2

    # neptune - track of your parameters
    PARAMS = {
        'Method': method,
        'Prefixes': prefixes,
        'Num Samples': num_samples,
        'Total Samples': total_samples,
        'Length': length,
        'Seed': 2,
        'Stepsize': 0.04,
        'Sample': True,
        'Num Iterations': num_iterations,
        'Gamma': 1,
        'GM Scale': 0.95,
        'KL Scale': 0.01,
    }

    # neptune - start an experiment
    neptune.init('guaguago/PPLM')
    neptune.create_experiment(name='Sentiment Control', params=PARAMS)

    # neptune - log PPL
    neptune.log_metric('PPL Pos', float('{:.3f}'.format(pos_ppl)))
    neptune.log_metric('PPL Neg', float('{:.3f}'.format(neg_ppl)))
    neptune.log_metric('PPL Mean', float('{:.3f}'.format(mean_ppl)))

    # neptune - log dist-1,2,3
    neptune.log_metric('Dist-1 Pos', float('{:.3f}'.format(pos_dist_1)))
    neptune.log_metric('Dist-2 Pos', float('{:.3f}'.format(pos_dist_2)))
    neptune.log_metric('Dist-3 Pos', float('{:.3f}'.format(pos_dist_3)))
    neptune.log_metric('Dist-1 Neg', float('{:.3f}'.format(neg_dist_1)))
    neptune.log_metric('Dist-2 Neg', float('{:.3f}'.format(neg_dist_2)))
    neptune.log_metric('Dist-3 Neg', float('{:.3f}'.format(neg_dist_3)))
    neptune.log_metric('Dist-1 Mean', float('{:.3f}'.format(mean_dist_1)))
    neptune.log_metric('Dist-2 Mean', float('{:.3f}'.format(mean_dist_2)))
    neptune.log_metric('Dist-3 Mean', float('{:.3f}'.format(mean_dist_3)))

    # neptune - log acc
    neptune.log_metric('Acc Pos', float('{:.3f}'.format(pos_acc)))
    neptune.log_metric('Acc Neg', float('{:.3f}'.format(neg_acc)))
    neptune.log_metric('Acc Mean', float('{:.3f}'.format(mean_acc)))

    # neptune - log samples
    [neptune.log_text('Pos Samples', s) for s in pos_samples]
    [neptune.log_text('Neg Samples', s) for s in neg_samples]

    neptune.stop()
