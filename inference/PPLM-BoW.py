from run_pplm import run_pplm_example

if __name__ == '__main__':

    # 1.Set prefixes
    prefixes = [
        # standard 15 prefixes
        'Once upon a time',
        # 'The book', 'The chicken',
        # 'The city', 'The country',
        # 'The horse', 'The lake', 'The last time', 'The movie', 'The painting',
        # 'The pizza', 'The potato', 'The president of the country', 'The road', 'The year is 1910.',
        # # # extra 35 prefixes
        # 'The article', 'I would like to', 'We should', 'In the future', 'The cat',
        # 'The piano', 'The walls', 'The hotel', 'The good news', 'The building',
        # 'The owner', 'Our house', 'Do you like', 'Her hair', 'The spider man',
        # 'The computer', 'My phone', 'The TV', 'The bus', 'Long long ago',
        # 'My daughter', 'The ice cream', 'This recipe', 'Most of us', 'The game',
        # 'The music', 'The show', 'The dress', 'In the evening', 'The traffic',
        # 'We usually', 'My mother', 'My dad', 'The meeting', 'My wife',
    ]

    # 2.Config Hyperparameters
    task = 'positive'
    sample_methods = 'PPLM-BoW'
    seed = 20
    num_samples_each_prefix = 1
    total_samples = len(prefixes) * num_samples_each_prefix
    num_iterations = 10
    step_size = 0.04  # control strength "dead dead ... dead"
    length = 30
    verbosity = 'quiet'

    # 3.Set Output File Path
    dst_file_name = '{},{},seed={},n={},m={}'.format(
        task, sample_methods, seed, total_samples, num_iterations)
    dst_file = 'automated_evaluation/generated_samples/{}/{}'.format(task, dst_file_name)

    # 4.Inference
    for prefix in prefixes:
        with open(dst_file, 'a') as file:
            run_pplm_example(
                bag_of_words='dead',
                cond_text=prefix,
                num_samples=num_samples_each_prefix,
                length=length,  # influence random
                seed=seed,
                stepsize=step_size,
                sample=True,
                num_iterations=num_iterations,
                gamma=1,
                gm_scale=0.95,
                kl_scale=0.01,
                verbosity=verbosity,
                file=file,
                sample_method=sample_methods,
            )
