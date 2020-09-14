if __name__ == '__main__':
    print(298 / 15)
    print(298 * 14)
    # Wilcoxon signed-rank test
    from scipy.stats import wilcoxon
    import statistics as stat

    # baseline: pplm-o
    acc_pplm = [60, 59.2, 58.1, 60.3, 60.1, 59.5, 59.2, 57.8, 60.2, 61.6]
    ppl_pplm = [46.97, 48.93, 51.24, 48.55, 49.38, 47.80, 48.55, 48.89, 48.3, 45.6]
    dist1_pplm = [0.205, 0.205, 0.211, 0.211, 0.199, 0.203, 0.204, 0.207, 0.201, 0.201]
    dist2_pplm = [0.582, 0.585, 0.592, 0.591, 0.572, 0.572, 0.582, 0.587, 0.581, 0.584]
    dist3_pplm = [0.804, 0.81, 0.811, 0.812, 0.797, 0.795, 0.806, 0.812, 0.804, 0.807]

    # baseline: vad-o
    acc_vad_o = [63, 61.5, 58.7, 58.3, 63, 59.6, 62.8, 60.4, 63.5, 63.3]
    ppl_vad_o = [45.177, 43.828, 43.097, 43.516, 42.349, 43.439, 41.91, 48.634, 45.582, 43.827]
    dist1_vad_o = [0.222, 0.219, 0.216, 0.215, 0.215, 0.221, 0.218, 0.227, 0.22, 0.216]
    dist2_vad_o = [0.64, 0.637, 0.632, 0.632, 0.629, 0.635, 0.629, 0.647, 0.632, 0.636]
    dist3_vad_o = [0.852, 0.849, 0.853, 0.846, 0.849, 0.849, 0.847, 0.86, 0.847, 0.855]

    # baseline: loss-o
    acc_loss_o = [61.5, 64.1, 63.2, 64, 63.4, 62.3, 62.8, 62.9, 66.1, 67]
    ppl_loss_o = [46.244, 45.607, 43.086, 45.787, 44.202, 47.298, 42.934, 49.487, 44.274, 43.531]
    dist1_loss_o = [0.214, 0.216, 0.214, 0.214, 0.214, 0.22, 0.216, 0.22, 0.213, 0.219]
    dist2_loss_o = [0.633, 0.633, 0.632, 0.633, 0.633, 0.637, 0.628, 0.642, 0.628, 0.637]
    dist3_loss_o = [0.849, 0.845, 0.852, 0.851, 0.849, 0.847, 0.847, 0.858, 0.843, 0.855]

    # # baseline: vad-u
    # acc_vad_u = [61.8, 58.1, 61.6, 61.7, 59.3, 61.6, 62.1, 60, 61.2, 58.5]
    # ppl_vad_u = [43.867, 42.912, 39.595, 42.87, 41.55, 42.198, 42.378, 44.127, 42.328, 39.725]
    # dist1_vad_u = [0.224, 0.229, 0.226, 0.225, 0.224, 0.229, 0.228, 0.228, 0.223, 0.226]

    # # baseline: loss-u
    # acc_loss = [62.5, 61.1, 63.6, 62.6, 63.1, 61, 63.6, 60.1, 64.1, 61.9]
    # ppl_loss = [44.27, 45.43, 42, 44.52, 43.55, 44.7, 43.57, 46.33, 44.02, 42.88]
    # dist1_loss = [0.225, 0.228, 0.223, 0.227, 0.222, 0.228, 0.223, 0.229, 0.221, 0.223]

    # # baseline: pplm-u
    # acc_pplm_u = [60.7, 60.4, 59.6, 60.4, 59.4, 61.3, 60, 59.8, 60.8, 62.9]
    # ppl_pplm_u = [45.498, 44.998, 42.269, 43.293, 40.282, 43.491, 44.547, 45.484, 42.381, 43.054]
    # dist1_pplm_u = [0.234, 0.233, 0.236, 0.233, 0.224, 0.231, 0.234, 0.236, 0.225, 0.226]

    print('10 seeds 1000 samples results by mean/sdv:')
    print('          acc,      ppl,       dist-1,        dist-2,          dist-3')
    print('pplm-o: {:.2f}/{:.2f}, {:.2f}/{:.2f}, {:.3f}/{:.3f}, {:.3f}/{:.3f}, {:.3f}/{:.3f}'.format(
        stat.mean(acc_pplm), stat.stdev(acc_pplm),
        stat.mean(ppl_pplm), stat.stdev(ppl_pplm),
        stat.mean(dist1_pplm), stat.stdev(dist1_pplm),
        stat.mean(dist2_pplm), stat.stdev(dist2_pplm),
        stat.mean(dist3_pplm), stat.stdev(dist3_pplm)
    ))
    print('vad-o: {:.2f}/{:.2f}, {:.2f}/{:.2f}, {:.3f}/{:.3f}, {:.3f}/{:.3f}, {:.3f}/{:.3f}'.format(
        stat.mean(acc_vad_o), stat.stdev(acc_vad_o),
        stat.mean(ppl_vad_o), stat.stdev(ppl_vad_o),
        stat.mean(dist1_vad_o), stat.stdev(dist1_vad_o),
        stat.mean(dist2_vad_o), stat.stdev(dist2_vad_o),
        stat.mean(dist3_vad_o), stat.stdev(dist3_vad_o)
    ))
    print('loss-o: {:.2f}/{:.2f}, {:.2f}/{:.2f}, {:.3f}/{:.3f}, {:.3f}/{:.3f}, {:.3f}/{:.3f}'.format(
        stat.mean(acc_loss_o), stat.stdev(ppl_loss_o),
        stat.mean(ppl_loss_o), stat.stdev(ppl_loss_o),
        stat.mean(dist1_loss_o), stat.stdev(dist1_loss_o),
        stat.mean(dist2_loss_o), stat.stdev(dist2_loss_o),
        stat.mean(dist3_loss_o), stat.stdev(dist3_loss_o),
    ))


    # print('pplm-u: {:.2f}/{:.2f}, {:.2f}/{:.2f}, {:.3f}/{:.3f}'.format(
    #     stat.mean(acc_pplm_u), stat.stdev(acc_pplm_u),
    #     stat.mean(ppl_pplm_u), stat.stdev(ppl_pplm_u),
    #     stat.mean(dist1_pplm_u), stat.stdev(dist1_pplm_u)
    # ))
    # print('vad-u: {:.2f}/{:.2f}, {:.2f}/{:.2f}, {:.3f}/{:.3f}'.format(
    #     stat.mean(acc_vad_u), stat.stdev(acc_vad_u),
    #     stat.mean(ppl_vad_u), stat.stdev(ppl_vad_u),
    #     stat.mean(dist1_vad_u), stat.stdev(dist1_vad_u)
    # ))
    # print('loss-u: {:.2f}/{:.2f}, {:.2f}/{:.2f}, {:.3f}/{:.3f}'.format(
    #     stat.mean(acc_loss), stat.stdev(acc_loss),
    #     stat.mean(ppl_loss), stat.stdev(ppl_loss),
    #     stat.mean(dist1_loss), stat.stdev(dist1_loss)
    # ))

    _, p_acc = wilcoxon(acc_pplm, acc_loss)
    print('acc p-value={}'.format(p_acc))
    _, p_ppl = wilcoxon(ppl_pplm, ppl_loss)
    print('dist p-value={}'.format(p_ppl))
    _, p_dist1 = wilcoxon(dist1_pplm, dist1_loss)
    print('dist p-value={}'.format(p_dist1))

    # interpret
    alpha = 0.05
    if p_acc > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')

    # compare samples

    # interpret
    alpha = 0.05
    if p_acc > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')

    # interpret
    alpha = 0.05
    if p_acc > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')

    # compare samples
