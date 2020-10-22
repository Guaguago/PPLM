from run_pplm import run_pplm_example
import time
import os


def output_file_path(task, seed, total_samples, m):
    dst_file_name = '{},PPLM-BoW,seed={},n={},m={}'.format(
        task, seed, total_samples, m)
    return 'automated_evaluation/generated_samples/{}/{}'.format(task, dst_file_name)


def generation(bag_of_words, seed, num_samples_each_prefix, length, step_size, num_iterations, verbosity, prefixes,
               file):
    for prefix in prefixes:
        with open(file, 'a') as file:
            run_pplm_example(
                bag_of_words=bag_of_words,
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
            )


if __name__ == '__main__':
    # 1.Set prefixes
    prefixes = [
        # standard 15 prefixes
        'Once upon a time',
        'The book', 'The chicken',
        'The city', 'The country',
        'The horse', 'The lake', 'The last time', 'The movie', 'The painting',
        'The pizza', 'The potato', 'The president of the country', 'The road', 'The year is 1910.',
        # # # extra 35 prefixes
        # 'The article', 'I would like to', 'We should', 'In the future', 'The cat',
        # 'The piano', 'The walls', 'The hotel', 'The good news', 'The building',
        # 'The owner', 'Our house', 'Do you like', 'Her hair', 'The spider man',
        # 'The computer', 'My phone', 'The TV', 'The bus', 'Long long ago',
        # 'My daughter', 'The ice cream', 'This recipe', 'Most of us', 'The game',
        # 'The music', 'The show', 'The dress', 'In the evening', 'The traffic',
        # 'We usually', 'My mother', 'My dad', 'The meeting', 'My wife',
    ]

    # 2.Common Hyperparameters
    seed = 1
    num_samples_each_prefix = 3
    total_samples = len(prefixes) * num_samples_each_prefix
    num_iterations = 10
    step_size = 0.03  # control strength "dead dead ... dead"
    length = 50
    verbosity = 'quiet'

    # 3.Positive Control With Specific Parameters Assigned
    start_time = time.time()
    output_file = output_file_path('positive', seed, total_samples, num_iterations)
    generation(
        bag_of_words='positive_words',
        seed=seed,
        num_samples_each_prefix=num_samples_each_prefix,
        num_iterations=num_iterations,
        step_size=step_size,
        length=length,
        verbosity=verbosity,
        prefixes=prefixes,
        file=output_file
    )
    time_lag = time.time() - start_time
    os.rename(output_file, '{},time={:.2f}'.format(output_file, time_lag))

    # 4.Negative Control With Specific Parameters Assigned
    start_time = time.time()
    output_file = output_file_path('negative', seed, total_samples, num_iterations)
    generation(
        bag_of_words='negative_words',
        seed=seed,
        num_samples_each_prefix=num_samples_each_prefix,
        num_iterations=num_iterations,
        step_size=step_size,
        length=length,
        verbosity=verbosity,
        prefixes=prefixes,
        file=output_file
    )
    time_lag = time.time() - start_time
    os.rename(output_file, '{},time={:.2f}'.format(output_file, time_lag))
