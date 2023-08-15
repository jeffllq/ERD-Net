import argparse


parser = argparse.ArgumentParser(description='proposed model')
parser.add_argument("--dataset", type=str, default='ICEWS14',  help="dataset to choose" )
parser.add_argument("--device-id", type=int, default=0,help="device id to use" )
parser.add_argument("--n-hidden", type=int, default=200, help="number of hidden units")
parser.add_argument("--dropout", type=float, default=0.2,help="dropout probability")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--weight-decay", type=float, default=0.00001, help="learning rate weight_decay")
parser.add_argument("--alpha", type=float, default=0, help="learning rate weight_decay")
parser.add_argument("--use-bias", type=int, default=0, help="learning rate weight_decay")

parser.add_argument("--freq-info", type=int, default=0, help="use freq info or not")
parser.add_argument("--train-history-len", type=int, default=2, help="history length")
# parser.add_argument("--test-history-len", type=int, default=3, help="history length")
parser.add_argument("--n-layers", type=int, default=2, help="number of propagation rounds")
parser.add_argument("--input-dropout", type=float, default=0.2, help="input dropout for decoder ")
parser.add_argument("--hidden-dropout", type=float, default=0.2,  help="hidden dropout for decoder")
parser.add_argument("--feat-dropout", type=float, default=0.2, help="feat dropout for decoder")

parser.add_argument("--max-epochs", type=int, default=20, help="number of maximum training epochs on each time step")
parser.add_argument("--valid-step", type=int, default=2, help="valid gap")
parser.add_argument("--mode", type=str, default='pretrain', help="train mode or test mode", choices=['pretrain', 'train','gene_train', 'test'])
# parser.add_argument("--verse", type=int, default=0, help="window size change or not")
parser.add_argument("--filter", type=int, default=0, help="filter metrics or not")
args = parser.parse_args()

