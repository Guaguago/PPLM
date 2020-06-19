from run_pplm import run_pplm_example


def generate_samples(prefixes, sample_methods, dir, method_label, sentiment_label, verbose):
    for prefix in prefixes:
        with open('{}/{}'.format(dir, method_label), 'a') as file:
            for method in sample_methods:
                run_pplm_example(
                    cond_text=prefix,
                    num_samples=3,
                    discrim='sentiment',
                    class_label=sentiment_label,
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
                    sample_method=method
                )

            # file.write('=' * 89)


if __name__ == '__main__':
    # Set inputs
    prefixes = ['Once upon a time', 'The book', 'The chicken', 'The city', 'The country', 'The horse', 'The lake',
                'The last time',
                'The movie', 'The painting', 'The pizza', 'The potato', 'The president of the country', 'The road',
                'The year is 1910.']

    sample_methods = [
        # 'perturbed',
        # 'vad',
        'vad_abs'
    ]

    # sentiment_lable = 'positive'
    sentiment_lable = 'positive'

    method_label = 'BC2'
    # dir = 'pplm/generated/positive'
    dir = '/Users/xuchen/core/pycharm/project/PPL/automated_evaluation/vad/samples/{}'.format(sentiment_lable)
    generate_samples(prefixes, sample_methods, dir, method_label, 'very_{}'.format(sentiment_lable), 'quiet')
