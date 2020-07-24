if __name__ == '__main__':
    # Wilcoxon signed-rank test
    from numpy.random import seed
    from numpy.random import randn
    from scipy.stats import wilcoxon

    # seed the random number generator
    seed(1)
    # generate two independent samples
    ppl1 = [49.003, 49.121, 46.401, 48.099, 50.251]
    ppl2 = [46.786, 46.396, 45.02, 47.603, 45.216]

    # compare samples
    _, p_ppl = wilcoxon(ppl1, ppl2)
    print('ppl p-value={}'.format(p_ppl))

    # interpret
    alpha = 0.05
    if p_ppl > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')

    acc1 = [60.3, 59.2, 59.8, 61.4, 61.9]
    acc2 = [64.1, 65.3, 63.3, 60.9, 63.2]
    # compare samples
    _, p_acc = wilcoxon(acc1, acc2)
    print('acc p-value={}'.format(p_acc))

    # interpret
    alpha = 0.05
    if p_acc > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')
