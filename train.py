import utilities
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--cuda', type=str, default = "cpu",help="cuda status", action = "store")
parser.add_argument('--path', type=str, default ="flowers", help="folder containing dataset", action="store")
parser.add_argument('--dropout',type=float, default=0.5)
parser.add_argument('--lr',type=float, default=0.001, help="learning rate")
parser.add_argument('--arch', type=str,default='vgg16',help="select the model architecture",action='store')
parser.add_argument('--epochs',type=int,default=10,help="number of epochs for training..",action='store')
parser.add_argument('--hlayers',type=int,default=130,help="Number of first hidden layer", action='store')
args = parser.parse_args()
#load the dataset
t_loader,v_loader,test_loader=utilities.get_datasets(args.path)
#create the new network
if args.cuda == "gpu" :
    cuda=True
else:
    cuda=False
model1,optim1,criter1=utilities.setup_network(arch=args.arch,cuda=cuda, dropout=args.dropout,num_first_hidden_layer=args.hlayers,lr=args.lr)
utilities.start_training_phase(model1,optim1,criter1,t_loader,v_loader,cuda,epochs=args.epochs)

# for now, run with 
# >> python train.py --cuda gpu --dropout 0.6 --lr 0.002 
# or
# >> python train.py 
# it will take the default values. 