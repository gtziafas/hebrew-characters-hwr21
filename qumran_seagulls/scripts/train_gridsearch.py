import os

learning_rates = [1e-04, 5e-04, 1e-03, 5e-03]
batch_sizes = [32, 64, 128]
weight_decays = [0.001, 0.005, 0.01, 0.05, 0.1]
dropout_rates = [[0.1, 0.1], [0.1, 0.2], [0.1, 0.25], [0.1, 0.33]]

for [dol, doh] in dropout_rates:
    for lr in learning_rates:
        for bs in batch_sizes:
            for wd in weight_decays:
                execstr = f"python3 -m qumran_seagulls.scripts.train_cnn_monkbrill -e 15 -early 2 -bs {bs} -lr {lr} -wd {wd} -kfold 10 -dol {dol} -doh {doh}"
                print(execstr)
                os.system("pwd")
                os.system(execstr)
