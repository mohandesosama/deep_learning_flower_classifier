# Imports here
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms, models

import json
from PIL import Image
from collections import OrderedDict

def get_datasets(path):
    data_dir = path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir ,transform = test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size =32,shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 32, shuffle = True)
    return trainloader,validationloader,testloader

#create the nueral network with vgg16 model
def setup_network(arch='vgg16',cuda=False, dropout=0.5, num_first_hidden_layer = 130,lr = 0.001):
    #set the model
    if arch=='vgg16':
        model = models.vgg16(pretrained=True)
        structure_input=25088
    elif arch=='densenet121':
        model = models.densenet121(pretrained=True)
        structure_input=1024
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        structure_input=9216
    else:
        print("error selecting the model ....")
    #set the model
    model = models.vgg16(pretrained=True)
    # implement the classifier
    for param in model.parameters():
        param.requires_grad = False   
    classifier = nn.Sequential(OrderedDict([
        ('inputs', nn.Linear(structure_input, num_first_hidden_layer)),
        ('relu1', nn.ReLU()),
        ('fc1', nn.Linear(num_first_hidden_layer, 100)),
        ('relu2',nn.ReLU()),
        ('fc2',nn.Linear(100,80)),
        ('relu3',nn.ReLU()),
        ('fc3',nn.Linear(80,102)),
        ('output', nn.LogSoftmax(dim=1)) ]))
    classifier.drop=nn.Dropout(dropout)
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr )
    if cuda:
        model.cuda()
    print("Model created successfully.....")
    return model , optimizer ,criterion 
# validate the constructed network
def validate_model(model,optimizer,criterion,validationloader,cuda=False):
    # validate the model with the calculated weights in backword/forward passes
    accuracy=0
    validation_loss = 0
    for k, (validation_inputs,validation_labels) in enumerate(validationloader):
        optimizer.zero_grad()
        # pass the validation phase to cuda for faster processing
        if cuda:
            validation_inputs, validation_labels = validation_inputs.cuda(), validation_labels.cuda() 
        with torch.no_grad():    
            validation_output = model.forward(validation_inputs)
            validation_loss = criterion(validation_output,validation_labels)
            ps = torch.exp(validation_output).data
            # matrix containing 1's for correct classfication and 0 for incorrect classification
            matching = (validation_labels.data == ps.max(1)[1])
            accuracy += matching.type_as(torch.FloatTensor()).mean()
    # Calculate validation loss and accuracy
    validation_loss = validation_loss / len(validationloader)
    accuracy = accuracy /len(validationloader)
    return validation_loss, accuracy
# training the newtork
def start_training_phase(model,optimizer,criterion,trainloader,validationloader,cuda=False,epochs=10):
    print("started the training process .... ")
    print_output_every = 5
    # transfer control to cuda
    if cuda:
        model.cuda()
    steps = 0
    for ee in range(epochs):
        running_loss = 0
        for dexdex, (inputs, labels) in enumerate(trainloader):
            if cuda:
                inputs,labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            # Forward and backward iterations
            outputs = model.forward(inputs)
            # calculate the error of the forward phase
            loss = criterion(outputs, labels)
            #backward
            loss.backward()
            #optimize for optimal weights in the backward phase
            optimizer.step()
            running_loss += loss.item()
            # increase loop steps
            steps += 1
            if steps % print_output_every == 0:
                model.eval()
                ## validation function
                validation_loss, accuracy = validate_model(model,optimizer,criterion,validationloader,cuda)
                   #print the results
                print("Epoch: {}/{}..... ".format(ee+1, epochs),"Model Loss: {:.3f}".format(running_loss/print_output_every),
                      "Validation Loss {:.3f}".format(validation_loss),"Accuracy: {:.3f}".format(accuracy))
                running_loss = 0
    print("Training phase is finished successfully")
# TODO: Do validation on the test set
def load_the_checkpoint(fname,is_cuda_available=False):
    if is_cuda_available:
        checkpoint = torch.load(fname)
    else:
        checkpoint = torch.load(fname, map_location ='cpu')
    #because of vgg16 model
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_first_hlayer = checkpoint['num_first_hidden_layer']
    model,_,_ = setup_network(cuda=is_cuda_available, num_first_hidden_layer=num_first_hlayer)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    print("Model loaded successfully ..")
    return model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)
    processessing = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) 
    return processessing(pil_image)
# TODO: Implement the code to predict the class from an image file
def predict(image_path, model, topk=5):   
    model.eval()
    model.cpu()    
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    with torch.no_grad():
        output = model.forward(img_torch.cpu())    
    return F.softmax(output.data,dim=1).topk(topk)
def get_categ_names(file_path):
    import json
    with open(file_path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name