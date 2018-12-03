import utilities
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--cuda', type=str, default = "cpu",help="cuda status", action = "store")
parser.add_argument('--fname', type=str, default = None, help ="checkpoint file",action = "store")
parser.add_argument('--jsonfile',type=str, default='cat_to_name.json',action="store")
args = parser.parse_args()
if args.cuda == "gpu" :
    cuda=True
else:
    cuda=False
model=utilities.load_the_checkpoint(fname=args.fname,is_cuda_available=cuda)
path ='flowers/test/1/image_06743.jpg'
probabilities = utilities.predict(path, model)
print(probabilities)
cat_to_names=utilities.get_categ_names(args.jsonfile)

