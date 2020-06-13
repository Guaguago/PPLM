from run_pplm import run_pplm_example

if __name__ == '__main__':
    prefix = 'My father'
    with open('demos/pos2neg', 'a') as file:
        file.write(
            '========================================================================================================================================================\n')
        run_pplm_example(
            cond_text=prefix,
            num_samples=1,
            discrim='sentiment',
            class_label='very_negative',
            length=50,
            stepsize=0.05,
            sample=True,
            num_iterations=5,
            gamma=1,
            gm_scale=0.9,
            kl_scale=0.02,
            verbosity='regular',
            file=file,
            generation_method='vad'

        )
