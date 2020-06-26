from run_pplm import run_pplm_example

if __name__ == '__main__':
    prefix = ['The orange', 'The spider man', 'my father']
    for p in prefix:
        with open('demos/religion', 'a') as file:
            file.write(
                '========================================================================================================================================================\n')
            file.write('【{}】\n'.format(p))
            run_pplm_example(
                cond_text=p,
                num_samples=1,
                bag_of_words='religion',
                length=50,
                stepsize=0.03,
                sample=True,
                num_iterations=3,
                window_length=5,
                gamma=1.5,
                gm_scale=0.95,
                kl_scale=0.01,
                verbosity='regular',
                file=file,
                generation_method='vad_abs'

            )
