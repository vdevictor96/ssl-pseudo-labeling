import torch
import os
from os.path import join as pjoin
from dataloader import get_cifar10, get_cifar100
from model.wrn import WideResNet
import torch.nn.functional as F
from utils import accuracy
from torch.utils.data import DataLoader

def test_cifar10(testdataset, filepath = "./path/to/model.pth.tar"):
    '''
    args: 
        testdataset : (torch.utils.data.Dataset)
        filepath    : (str) The path to the model file that is saved
    returns : (torch.Tensor) logits of the testdataset with shape 
                [num_samples, 10]. Apply softmax to the logits
    
    Description:
        This function loads the model given in the filepath and returns the 
        logits of the testdataset which is a torch.utils.data.Dataset. You can
        save the arguments needed to load the models (e.g. width, depth etc) 
        with the model file. Assume testdataset is like CIFAR-10. Test this
        function with the testdataset returned by get_cifar10()
    '''
    # CREATE LOADER 
   
    test_loader = DataLoader(testdataset,
                             batch_size=64,
                             shuffle=False,
                             num_workers=1)
    
    
    # RETRIEVE MODEL
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelpath = torch.load(pjoin(filepath))
    model = WideResNet(modelpath['model_depth'],
                       modelpath['num_classes'], widen_factor=modelpath['model_width'], dropRate=modelpath['drop_rate'])
    model = model.to(device)
    model.load_state_dict(modelpath['model_state_dict'])

    # RETURN SOFTMAX
    model.eval()
    outputs = torch.empty((0, 10)).to(device)
    for x_test, _ in test_loader:
        with torch.no_grad():
            x_test = x_test.to(device)
            output_test = model(x_test)
            softmax_test = F.softmax(output_test, dim=1)
            outputs = torch.cat((outputs, softmax_test))
    return outputs
    '''
    model.eval()
    x_test, _ = testdataset[:]
    x_test = x_test.to(device)
    outputs_test = model(x_test)
    softmax_test = F.softmax(outputs_test, dim=1)
    return softmax_test
    '''
    

def test_cifar100(testdataset, filepath="./path/to/model.pth.tar"):
    '''
    args: 
        testdataset : (torch.utils.data.Dataset)
        filepath    : (str) The path to the model file that is saved
    returns : (torch.Tensor) logits of the testdataset with shape 
                [num_samples, 100]. Apply softmax to the logits
    
    Description:
        This function loads the model given in the filepath and returns the 
        logits of the testdataset which is a torch.utils.data.Dataset. You can
        save the arguments needed to load the models (e.g. width, depth etc) 
        with the model file. Assume testdataset is like CIFAR-100. Test this
        function with the testdataset returned by get_cifar100()
    '''
    # CREATE LOADER 
   
    test_loader = DataLoader(testdataset,
                             batch_size=64,
                             shuffle=False,
                             num_workers=1)
    
    
    # RETRIEVE MODEL
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelpath = torch.load(pjoin(filepath))
    model = WideResNet(modelpath['model_depth'],
                       modelpath['num_classes'], widen_factor=modelpath['model_width'], dropRate=modelpath['drop_rate'])
    model = model.to(device)
    model.load_state_dict(modelpath['model_state_dict'])

    # RETURN SOFTMAX
    model.eval()
    outputs = torch.empty((0, 100)).to(device)
    for x_test, _ in test_loader:
        with torch.no_grad():
            x_test = x_test.to(device)
            output_test = model(x_test)
            softmax_test = F.softmax(output_test, dim=1)
            outputs = torch.cat((outputs, softmax_test))
    return outputs