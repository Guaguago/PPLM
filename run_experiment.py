import neptune
from automated_evaluation.cal_ppl import cal_ppl
from automated_evaluation.cal_dist import cal_dist
from automated_evaluation.cal_acc import cal_acc


def get_samples(src):
    with open(src, 'r') as f:
        return f.read().split('<|endoftext|>')[1:]


def startswith(string, prefixes):
    for prefix in prefixes:
        if string.startswith(prefix):
            return True
    return False


def drop(src):
    samples = get_samples(src)
    new_samples = []
    for sample in samples:
        if startswith(sample, prefixes):
            new_samples.append(sample)
        else:
            new_samples[len(new_samples) - 1] += ' ' + sample
    return new_samples


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

    neg_src = 'automated_evaluation/generated_samples/new/seed=2,negative,name=BC_VAD,samples=75,itrs=10,vad_t=0.01, lambda=1000.0, pos_t=0.9, neg_t=0.1,changes=12.693333333333333'
    pos_src = 'automated_evaluation/generated_samples/new/seed=2,positive,name=BC_VAD,samples=75,itrs=10,vad_t=0.01, lambda=1000.0, pos_t=0.9, neg_t=0.1,changes=17.573333333333334'

    # Only for VAD_LOSS
    LAMBDA = 1000.0
    pos_threshold = 0.9
    neg_threshold = 0.1
    neg_changes = 12.69
    pos_changes = 17.57
    method = 'vad-loss'
    seed = 2
    num_samples = 5
    total_samples = len(prefixes) * num_samples

    vad_threshold = 0.01
    length = 50
    num_iterations = 10

    # obtain samples

    pos_samples = drop(pos_src)
    neg_samples = drop(neg_src)
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
        'method': method,
        'prefixes': prefixes,
        'num_samples': num_samples,
        'total_samples': total_samples,
        'length': length,
        'seed': seed,
        'stepsize': 0.04,
        'sample': True,
        'num_iterations': num_iterations,
        'gamma': 1,
        'gm_scale': 0.95,
        'kl_scale': 0.01,
        'vad_threshold': vad_threshold,
        'pos_threshold': pos_threshold,
        'neg_threshold': neg_threshold,
        'lambda': LAMBDA

    }

    # neptune - start an experiment
    neptune.init('guaguago/change-or-not')
    neptune.create_experiment(name='sentiment-control', params=PARAMS)

    # neptune - log PPL
    neptune.log_metric('pos-ppl', float('{:.3f}'.format(pos_ppl)))
    neptune.log_metric('neg-ppl', float('{:.3f}'.format(neg_ppl)))
    neptune.log_metric('mean-ppl', float('{:.3f}'.format(mean_ppl)))

    # neptune - log dist-1,2,3
    neptune.log_metric('pos-dist-1', float('{:.3f}'.format(pos_dist_1)))
    neptune.log_metric('pos-dist-2', float('{:.3f}'.format(pos_dist_2)))
    neptune.log_metric('pos-dist-3', float('{:.3f}'.format(pos_dist_3)))
    neptune.log_metric('neg-dist-1', float('{:.3f}'.format(neg_dist_1)))
    neptune.log_metric('neg-dist-2', float('{:.3f}'.format(neg_dist_2)))
    neptune.log_metric('neg-dist-3', float('{:.3f}'.format(neg_dist_3)))
    neptune.log_metric('mean-dist-1', float('{:.3f}'.format(mean_dist_1)))
    neptune.log_metric('mean-dist-2', float('{:.3f}'.format(mean_dist_2)))
    neptune.log_metric('mean-dist-3', float('{:.3f}'.format(mean_dist_3)))

    # neptune - log acc
    neptune.log_metric('pos-acc', float('{:.3f}'.format(pos_acc)))
    neptune.log_metric('neg-acc', float('{:.3f}'.format(neg_acc)))
    neptune.log_metric('mean-acc', float('{:.3f}'.format(mean_acc)))

    # neptune - log changes
    neptune.log_metric('pos-changes', pos_changes)
    neptune.log_metric('neg-changes', neg_changes)
    neptune.log_metric('mean-changes', float('{:.3f}'.format((pos_changes + neg_changes) / 2)))

    # neptune - log samples
    [neptune.log_text('pos-samples', s) for s in pos_samples]
    [neptune.log_text('neg-samples', s) for s in neg_samples]

    neptune.stop()
