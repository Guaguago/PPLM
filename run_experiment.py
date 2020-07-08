import neptune
from automated_evaluation.cal_ppl import cal_ppl

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

    # calculate PPL and samples
    pos_ppl, pos_samples = cal_ppl(pos_src)
    neg_ppl, neg_samples = cal_ppl(neg_src)
    mean_ppl = (pos_ppl + neg_ppl) / 2

    # neptune - log PPL
    neptune.log_metric('PPL-Pos', pos_ppl)
    neptune.log_metric('PPL-Neg', neg_ppl)
    neptune.log_metric('PPL-Mean', mean_ppl)

    # neptune - log samples
    [neptune.log_text('Pos Samples', s) for s in pos_samples]
    [neptune.log_text('Neg Samples', s) for s in neg_samples]

    neptune.stop()
