# parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
#                       help='Comma separated list of number of units in each hidden layer')
#   parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
#                       help='Learning rate')
#   parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
#                       help='Number of steps to run trainer.')
#   parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
#                       help='Batch size to run trainer.')
#   parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
#                         help='Frequency of evaluation on the test set')
#   parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
#                       help='Directory for storing input data')


# python train_mlp_pytorch.py > pytorch_results.txt
# python train_mlp_pytorch.py --dnn_hidden_units="100, 100" >> pytorch_results.txt
# python train_mlp_pytorch.py --dnn_hidden_units="100, 100, 100" >> pytorch_results.txt


# python train_mlp_pytorch.py --learning_rate=1e-2 >> pytorch_results.txt
# python train_mlp_pytorch.py --learning_rate=1e-3 >> pytorch_results.txt
# python train_mlp_pytorch.py --learning_rate=1e-4 >> pytorch_results.txt

# python train_mlp_pytorch.py --batch_size=100 >> pytorch_results.txt
# python train_mlp_pytorch.py --batch_size=300 >> pytorch_results.txt

# python train_mlp_pytorch.py --optimizer='adam' >> pytorch_results.txt

python train_mlp_pytorch.py --dnn_hidden_units="300, 200, 100, 100, 100, 100" --batch_size=300 --max_steps=4000 --learning_rate=1e-2 >> pytorch_results.txt

