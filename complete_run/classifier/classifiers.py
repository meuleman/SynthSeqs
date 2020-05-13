



if __name__ == "__main__":
    ### ALL ###
    lrs = np.arange(0.001, 0.009, 0.0008)
    beta1s = np.arange(0.7, 0.9, 0.1)
    beta2s = np.arange(0.69, 1.0, 0.1)
    # n_filters = range(5, 155, 10)


    ### MODEL PARAMS ###
    model = conv_net
    filters = [8, 16, 32, 64]
    pool_sizes = [5, 2, 10]
    fully_connecteds = [50, 100, 150]
    drops = [0.1, 0.2]

    model_param_set = product(filters,
                              pool_sizes,
                              fully_connecteds,
                              drops)

    print('n_filters\tpool_size\tn_fc\tdrop\ttrain_loss\tval_loss')
    for model_params in model_param_set:
        trainer = classifier_trainer(10,
                                     256,
                                     0.0018,
                                     model_params,
                                     model,
                                     b1=0.9,
                                     b2=0.99)
        trained = trainer.train()

        n_filters, pool_size, n_fc, drop = model_params
        print(str(n_filters), end="\t")
        print(str(pool_size), end="\t")
        print(str(n_fc), end="\t")
        print(str(drop), end="\t")
        print(str(trainer.train_hist[-1]), end="\t")
        print(str(trainer.train_hist_test[-1]))
        # print('Train loss: {}'.format(trainer.train_hist[-1]), end="\t")
        # print('Validation loss: {}'.format(trainer.train_hist_test[-1]))
    #model = trainer.train()
    #torch.save(model.state_dict(), "/home/pbromley/SynthSeqs/CompleteRun/saved_models/classifiers/large.pth")
