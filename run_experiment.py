import neptune
from automated_evaluation.cal_ppl import cal_ppl
from automated_evaluation.cal_dist import cal_dist

if __name__ == '__main__':
    prefixes = [
        'Once upon a time',
        'The book', 'The chicken', 'The city', 'The country', 'The horse', 'The lake',
        'The last time',
        'The movie', 'The painting', 'The pizza', 'The potato', 'The president of the country', 'The road',
        'The year is 1910.'
    ]

    pos_src = 'data/test/generated_samples/paper/bc_pos(2_150_10)'
    neg_src = 'data/test/generated_samples/paper/bc_neg(2_150_10)'

    method = 'BC'

    # calculate PPL and samples
    pos_ppl, pos_samples = cal_ppl(pos_src)
    neg_ppl, neg_samples = cal_ppl(neg_src)
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

    # neptune - track of your parameters
    PARAMS = {
        'Method': method,
        'Prefixes': prefixes,
        'Num Samples': 3,
        'Total Samples': 150,
        'Length': 50,
        'Seed': 2,
        'Stepsize': 0.04,
        'Sample': True,
        'Num Iterations': 10,
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
    neptune.log_metric('Dist-1 Pos',  float('{:.3f}'.format(pos_dist_1)))
    neptune.log_metric('Dist-2 Pos',  float('{:.3f}'.format(pos_dist_2)))
    neptune.log_metric('Dist-3 Pos',  float('{:.3f}'.format(pos_dist_3)))
    neptune.log_metric('Dist-1 Neg',  float('{:.3f}'.format(neg_dist_1)))
    neptune.log_metric('Dist-2 Neg',  float('{:.3f}'.format(neg_dist_2)))
    neptune.log_metric('Dist-3 Neg',  float('{:.3f}'.format(neg_dist_3)))
    neptune.log_metric('Dist-1 Mean',  float('{:.3f}'.format(mean_dist_1)))
    neptune.log_metric('Dist-2 Mean',  float('{:.3f}'.format(mean_dist_2)))
    neptune.log_metric('Dist-3 Mean',  float('{:.3f}'.format(mean_dist_3)))

    # neptune - log samples
    [neptune.log_text('Pos Samples', s) for s in pos_samples]
    [neptune.log_text('Neg Samples', s) for s in neg_samples]

    neptune.stop()
