import unittest
from run_pplm import run_pplm_example


class TestMethods(unittest.TestCase):
    def test_BC_untached(self):
        with open('BC/output', 'w') as file:
            run_pplm_example(
                cond_text='Once upon a time',
                num_samples=1,
                discrim='sentiment',
                class_label=3,
                length=5,  # influence random
                seed=0,
                stepsize=0.05,
                sample=True,
                num_iterations=3,
                gamma=1,
                gm_scale=0.9,
                kl_scale=0.02,
                verbosity='quiet',
                file=file,
                sample_method='BC'
            )
        with open('BC/output', 'r') as file:
            output = file.read()
        with open('BC/known_output', 'r') as file:
            known_output = file.read()
        self.assertEqual(output, known_output)


unittest.main()