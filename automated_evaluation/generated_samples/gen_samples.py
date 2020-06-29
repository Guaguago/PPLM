from run_pplm import run_pplm_example


def generate_samples(prefixes, length, sample_methods, sentiment_label, verbose, suffix, seed=0):
    for method_name in sample_methods:
        output = '{}/{}{}'.format(sentiment_label, method_name, suffix)
        for prefix in prefixes:
            with open(output, 'a') as file:
                run_pplm_example(
                    cond_text=prefix,
                    num_samples=3,
                    discrim='sentiment',
                    class_label='very_{}'.format(sentiment_label),
                    length=length,  # influence random
                    seed=seed,
                    stepsize=0.04,
                    sample=True,
                    num_iterations=10,
                    gamma=1,
                    gm_scale=0.95,
                    kl_scale=0.01,
                    verbosity=verbose,
                    file=file,
                    sample_method=method_name
                )

        # file.write('=' * 89)


if __name__ == '__main__':
    prefixes = ['Once upon a time', 'The book', 'The chicken', 'The city', 'The country', 'The horse', 'The lake',
                'The last time',
                'The movie', 'The painting', 'The pizza', 'The potato', 'The president of the country', 'The road',
                'The year is 1910.']


    SEED = 2
    # single
    sentiment_label = [
        'positive',
        # 'negative'
    ]

    # multiple
    sample_methods = [
        'B',
        # 'BC',
        # 'BC_VAD',
        # 'BC_VAD_ABS'
    ]

    suffix = '(2_45_10)'

    generate_samples(prefixes, 50, sample_methods, sentiment_label[0], 'quiet', suffix, seed=SEED)
