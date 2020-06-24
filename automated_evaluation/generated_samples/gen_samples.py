from run_pplm import run_pplm_example


def generate_samples(prefixes, sample_methods, sentiment_label, verbose):
    for method_name in sample_methods:
        output = '{}/{}'.format(sentiment_label, method_name)
        with open(output, 'a') as file:
            for prefix in prefixes:
                run_pplm_example(
                    cond_text=prefix,
                    num_samples=3,
                    discrim='sentiment',
                    class_label='very_'.format(sentiment_label),
                    length=50,  # influence random
                    seed=0,
                    stepsize=0.05,
                    sample=True,
                    num_iterations=3,
                    gamma=1,
                    gm_scale=0.9,
                    kl_scale=0.02,
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

    # single
    sentiment_label = [
        'positive',
        # 'negative'
    ]
    

    # multiple
    sample_methods = [
        # 'BC',
        'BC_VAD',
        # 'BC_VAD_ABS'
    ]

    generate_samples(prefixes, sample_methods, sentiment_label[0], 'regular')
