import argparse
# arg for transformer work on "MAGAZINE"

def get_preprocess_args():
    parser = get_parser()
    add_dataset_args(parser)
    return parser.parse_args()

def get_trainning_args():
    parser = get_parser()
    add_model_args(parser)
    add_dataset_args(parser)
    add_optimization_args(parser)
    add_log_args(parser)
    return parser.parse_args()

def get_testing_args():
    return get_trainning_args()

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true', help='use CPU instead of CUDA')
    return parser


def add_model_args(parser):
    group = parser.add_argument_group('Model')
    group.add_argument('--d_model', default=512, help='') # 
    group.add_argument('--d_feedforward', default=2048, help='Dimension of the feedforward network model in nn.TransformerEncoder')
    group.add_argument('--common_vocab', default=6, help='Number of nn.TransformerEncoderLayer')
    group.add_argument('--n_encoder_layers', default=6, help='Number of nn.TransformerEncoderLayer')
    group.add_argument('--n_decoder_layers', default=6, help='Number of nn.TransformerDecoderLayer')
    group.add_argument('--n_heads', default=5, help='Number of heads in the multi-head attention models')
    group.add_argument('--dropout', default=0.1, help='Value of dropout')
    group.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), help="Type of positional embedding to use on top of the image features")
    
    group.add_argument('--clip', default=0.8, help='')
    # backbone model
    group.add_argument('--lr_backbone', default=1e-5, type=float)
    group.add_argument('--backbone', default='resnet50', type=str, help="Name of the convolutional backbone to use")


def add_optimization_args(parser):
    group = parser.add_argument_group('Optimization')
    group.add_argument('--n_epochs', default=50, help='')
    group.add_argument('--n_warmup_epochs',default=10,help='How many fist epochs are used to warm up')
    group.add_argument('--batch_size', default=12, help='')
    group.add_argument('--learning_rate', default=1e-4, help='')

def add_dataset_args(parser):
    group = parser.add_argument_group('Data process')
    group.add_argument('--buffer', default='./dataset/MAGAZINE/buffer/', help='')
    group.add_argument('--img_folder', default='./dataset/MAGAZINE/images/', help='')
    group.add_argument('--annotation_folder', default='./dataset/MAGAZINE/layoutdata/annotations', help='')
    group.add_argument('--grid_width', default=45, help='')
    group.add_argument('--grid_height', default=60, help='') # 225*300(div 5->45*60), affect size of 'vocab_size'
    group.add_argument('--input_size',default=(225,300), help='(width,height)')
    group.add_argument('--n_classes',default=5, help='')

def add_log_args(parser):
    group = parser.add_argument_group('Train log')
    group.add_argument('--log_root', default='./experiment/',help='Root folder for saving model and log')
    group.add_argument('--log_interval', default=30, help='')
    group.add_argument('--graph_size', default=(9,12), help='')