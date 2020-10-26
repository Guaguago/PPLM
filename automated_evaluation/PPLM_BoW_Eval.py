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
        'The article', 'I would like to', 'We should', 'In the future', 'The cat',
        'The piano', 'The walls', 'The hotel', 'The good news', 'The building',
        'The owner', 'Our house', 'Do you like', 'Her hair', 'The spider man',
        'The computer', 'My phone', 'The TV', 'The bus', 'Long long ago',
        'My daughter', 'The ice cream', 'This recipe', 'Most of us', 'The game',
        'The music', 'The show', 'The dress', 'In the evening', 'The traffic',
        'We usually', 'My mother', 'My dad', 'The meeting', 'My wife',
    ]

    pos_src = 'automated_evaluation/generated_samples/positive/positive,PPLM-BoW,BoW=positive_words,seed=1,n=500,m=3,time=8028.38'
    neg_src = 'automated_evaluation/generated_samples/negative/negative,PPLM-BoW,BoW=negative_words,seed=1,n=500,m=3,time=8053.16'

    seed = 1
    step_size = 0.01
    bow = 'wordlist'
    baseline_name = 'PPLM-BoW'
    num_samples_per_prefix = 10
    total_samples = len(prefixes) * num_samples_per_prefix * 2

    # vad_threshold = 0.01
    length = 50
    num_iterations = 3
    inference_time_per_word = (8028.38 + 8053.16) / total_samples / length
    # obtain samples

    pos_samples = drop(pos_src)
    neg_samples = drop(neg_src)
    assert len(pos_samples) == total_samples / 2
    assert len(neg_samples) == total_samples / 2

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
        'baseline': baseline_name,
        'prefixes': prefixes,
        'bow': bow,
        'total_samples': total_samples,
        'seed': seed,
        'num_samples_per_prefix': num_samples_per_prefix,
        'length': length,
        'step_size': 0.01,
        'sample': True,
        'num_iterations': num_iterations,
        'gamma': 1.5,
        'gm_scale': 0.9,
        'kl_scale': 0.01,
    }

    # neptune - start an experiment
    neptune.init('guaguago/bow-comparison')
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

    # neptune - log inference time
    neptune.log_metric('inference_time', float('{:.2f}'.format(inference_time_per_word)))

    # neptune - log samples
    [neptune.log_text('pos-samples', s) for s in pos_samples]
    [neptune.log_text('neg-samples', s) for s in neg_samples]

    neptune.stop()
