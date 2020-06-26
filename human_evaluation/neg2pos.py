from run_pplm import run_pplm_example

if __name__ == '__main__':

    # Set inputs
    prefix = 'The horse'
    sample_methods = [
        'perturbed',
        # 'vad_abs',
        # 'vad_abs'
    ]

    with open('demos/neg2pos', 'a') as file:
        for m in sample_methods:
            file.write(
                '\n================= "{}" to BC by {}) =================\n'.format(prefix, m))
            run_pplm_example(
                cond_text=prefix,
                num_samples=1,
                discrim='sentiment',
                class_label='very_positive',
                length=50,  # influence random
                seed=0,
                stepsize=0.05,
                sample=True,
                num_iterations=3,
                gamma=1,
                gm_scale=0.9,
                kl_scale=0.02,
                verbosity='quiet',
                file=file,
                sample_method=m
            )

            file.write('=' * 89)
