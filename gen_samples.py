from run_pplm import run_pplm_example
import statistics as stat


def generate_samples(prefixes, num_samples, length, method_name, sentiment_label, verbose, num_iterations,
                     vad_loss_params=None, seed=0, vad_threshold=0.01):
    total_samples = len(prefixes) * num_samples
    file_name = 'seed={},{},name={},samples={},itrs={},vad_t={}'.format(seed, sentiment_label, method_name,
                                                                        total_samples,
                                                                        num_iterations, vad_threshold)
    if vad_loss_params:
        file_name += ', lambda={}, pos_t={}, neg_t={}'.format(vad_loss_params['lambda'],
                                                              vad_loss_params['pos_threshold'],
                                                              vad_loss_params['neg_threshold'])

    output = 'automated_evaluation/generated_samples/{}/{}'.format(sentiment_label, file_name)

    word_changes_list = []
    for prefix in prefixes:
        with open(output, 'a') as file:
            word_changes = run_pplm_example(
                cond_text=prefix,
                num_samples=num_samples,
                discrim='sentiment',
                class_label='very_{}'.format(sentiment_label),
                length=length,  # influence random
                seed=seed,
                stepsize=0.04,
                sample=True,
                num_iterations=num_iterations,
                gamma=1,
                gm_scale=0.95,
                kl_scale=0.01,
                verbosity=verbose,
                file=file,
                sample_method=method_name,
                vad_loss_params=vad_loss_params,
                vad_threshold=vad_threshold,
            )
            word_changes_list.append(word_changes)

    with open(output, 'a') as file:
        file.write('<|endoftext|>' + '' + '{}'.format(stat.mean(word_changes_list)))


if __name__ == '__main__':
    prefixes = [
        # standard 15 prefixes
        'Once upon a time',
        'The book', 'The chicken', 'The city', 'The country',
        'The horse', 'The lake', 'The last time', 'The movie', 'The painting',
        'The pizza', 'The potato', 'The president of the country', 'The road', 'The year is 1910.',
        # extra 35 prefixes
        # 'The article', 'I would like to', 'We should', 'In the future', 'The cat',
        # 'The piano', 'The walls', 'The hotel', 'The good news', 'The building',
        # 'The owner', 'Our house', 'Do you like', 'Her hair', 'The spider man',
        # 'The computer', 'My phone', 'The TV', 'The bus', 'Long long ago',
        # 'My daughter', 'The ice cream', 'This recipe', 'Most of us', 'The game',
        # 'The music', 'The show', 'The dress', 'In the evening', 'The traffic',
        # 'We usually', 'My mother', 'My dad', 'The meeting', 'My wife',
    ]

    SEED = 2
    num_iterations = 5
    num_samples = 1
    # multiple
    sample_methods = [
        # 'B',
        # 'BC',
        'BC_VAD',
        # 'BC_VAD_MAX',
        # 'BC_VAD_ABS'
    ]

    vad_loss_params = {
        'lambda': 0.5,
        'pos_threshold': 0.6,
        'neg_threshold': 0.2,
    }

    generate_samples(prefixes, num_samples, 10, sample_methods[0], 'negative', 'quiet', num_iterations=num_iterations,
                     seed=SEED, vad_loss_params=vad_loss_params)
    generate_samples(prefixes, num_samples, 50, sample_methods[0], 'positive', 'quiet', num_iterations=num_iterations,
                     seed=SEED, vad_loss_params=vad_loss_params)
