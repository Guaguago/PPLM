from run_pplm import run_pplm_example

if __name__ == '__main__':
    prefix = 'My father died'
    with open('demos/neg2pos', 'a') as file:
        file.write(
            '========================================================================================================================================================\n')
        run_pplm_example(
            cond_text=prefix,
            num_samples=1,
            discrim='sentiment',
            class_label='very_positive',
            length=10,
            stepsize=0.05,
            sample=True,
            num_iterations=3,
            gamma=1,
            gm_scale=0.9,
            kl_scale=0.02,
            verbosity='regular',
            file=file,
            generation_method='vad'
        )
