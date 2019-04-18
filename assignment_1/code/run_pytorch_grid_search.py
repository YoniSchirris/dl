import subprocess

DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
OPTIMIZER_DEFAULT = 'sgd'

dnns = ['\"500, 500, 300\"', '\"500, 300, 100\"']
lrs = ['1e-2', '1e-3', '1e-4']
stepss = [2000, 3000, 4000]
batchs = [200, 400]
optims = ['\"sgd\"', '\"adam\"']
regs = ['1e-2', '1e-3']

for lr in lrs:
    for steps in stepss:
        print(
            #     "python train_mlp_pytorch.py --dnn_hidden_units={} --learning_rate={} --max_steps={} --batch_size={} --optimizer={} --regularizer={} >> pytorch_results_bali.txt".format(
            #         dnn, lr, steps, batch, optim, reg))
            # bashCommand = "python train_mlp_pytorch.py --dnn_hidden_units={} --learning_rate={} --max_steps={} --batch_size={} --optimizer={} --regularizer={} >> pytorch_results_bali.txt".format(
            #     dnn, lr, steps, batch, optim, reg)
            # # subprocess.call(bashCommand, shell=True)
            "python train_mlp_pytorch.py --learning_rate={} --max_steps={}".format(
                lr, steps))
        bashCommand = "python train_mlp_pytorch.py --learning_rate={} --max_steps={}".format(
            lr, steps)
        subprocess.call(bashCommand, shell=True)
