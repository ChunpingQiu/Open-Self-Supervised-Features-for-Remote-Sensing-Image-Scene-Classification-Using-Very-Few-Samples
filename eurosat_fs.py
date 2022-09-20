import torch
print("Torch version:", torch.__version__)
import clip
from PIL import Image
from tqdm import tqdm
import numpy as np
import os

from torchvision import datasets, transforms
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from util.dataloder_path import Dataloader_path_list_MB
from csv import writer
import argparse

from util.mis import processMB

from torchgeo.transforms import AugmentationSequential
import torchvision.transforms as transforms

import torch
from util.similarity import cosine_metric, euclidean_metric

def get_args_parser():

    parser = argparse.ArgumentParser('metric learning for image scene classification', add_help=False)

    # Model parameters
    parser.add_argument('--model_name', default='', type=str,
                        help='Name of model to train')
    parser.add_argument('--similar', default='', type=str,
                        help='similar')                        

    parser.add_argument('--root', default=" ", type=str,
                        help='path to data')
    parser.add_argument('--dataset', default='ucm', type=str,
                        help='data to test')

    parser.add_argument('--shot', default=3, type=int,
                        help='number of samples')
    parser.add_argument('--img_size', default=224, type=int,
                        help='input size of images')                        
                
    ##############################
    parser.add_argument('--cfg', type=str, required=False, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    args = parser.parse_args()

    config = []

    return args, config



mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
    
def main(args, config):
      
    augmentations = AugmentationSequential(
            transforms.Resize(args.img_size, interpolation=BICUBIC),
            transforms.Normalize(mean, std),         
            data_keys=["image"],
            ) 

    dataset=args.dataset
    shot=args.shot
    root=args.root

    folder=root+dataset#

    if not os.path.exists(folder):
        print("dataset folder not exist! ", folder)
        os._exit()

    classes = os.listdir(folder)
    if len(classes) == 0:
        print('The folder is empty')
        os._exit()

    classes=sorted(classes)
    print("class folders: ", classes)

    ####################################
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    
    #select the model
    model, preprocess = clip.load(args.model_name, device=device)

    # print(preprocess)
    ###########################################


    def fewshot_classifier(classes, shot, root, datasets, bands):

        folder=root+dataset#

        with torch.no_grad():
            fewshot_weights = []

            for idx_class in np.arange(len(classes)):#processing per class

                classname=classes[idx_class]

                f=open(root+"/experiment_split/"+dataset+"_support_p1_"+str(idx_class)+"_"+classname+".txt","r")
                lines = f.read().split('\n')

                for idx , line in enumerate(lines):# select labels
                    
                    if idx==shot:
                        break
                                        
                    img=folder+"/"+classname+"/"+line
                    imageTmp = processMB(img, bands, augmentations)


                    if idx==0:
                        image=[imageTmp]
                    else:
                        image.append(imageTmp)
                f.close()

                # print(image.type)
                image = torch.cat(image, 0)
                # print(image.shape)

                class_embeddings = model.encode_image(image.to(device)) 

                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()

                fewshot_weights.append(class_embedding)
            fewshot_weights = torch.stack(fewshot_weights, dim=1).cuda()

        return fewshot_weights

    ##############################################
    
    id=0
    #different band combinations for fusion
    for bands in [[3,2,1],[2,3,7],[2,3,4],[5,6,7],[7, 8,11], [1,11,12], [0,9,10]]:
        print(bands)
        outputAll=np.zeros((0))
        probsAll=np.zeros((0,len(classes)))
        
        if id==0:
            targetAll=np.zeros((0))
            
        with torch.no_grad():
            
            fewshot_weights = fewshot_classifier(classes, shot, root, dataset, bands)
            # print(fewshot_weights.shape)torch.Size([768, 21])

            for idx_class in np.arange(len(classes)):

                classname=classes[idx_class]
                
                datas = Dataloader_path_list_MB(root, dataset, idx_class, classname, augmentations, bands)
                data_loader = torch.utils.data.DataLoader(dataset=datas,
                                            num_workers=16,
                                            batch_size=256, shuffle=False)
                for image, target in data_loader:

                    image_features = model.encode_image(image.to(device)) 

                    image_features /= image_features.norm(dim=-1, keepdim=True)

                    # print(image_features.shape, fewshot_weights.shape)
                    if args.similar=='euclidean_metric':
                        logits_per_image = 100. * euclidean_metric(image_features, fewshot_weights) 
                    elif args.similar=='cosine_metric':
                        logits_per_image = 100. * cosine_metric(image_features, fewshot_weights)
                    else:
                        logits_per_image = 100. * image_features @ fewshot_weights


                    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                                        
                    output = np.uint8(np.argmax(probs, axis=1))
                    # print(output,targets)
                    outputAll = np.append(outputAll, output.flatten(), axis=0)
                    probsAll = np.append(probsAll, probs, axis=0)
                    
                    if id==0:
                        targetAll = np.append(targetAll, target.flatten(), axis=0)
                    

        accuracy = accuracy_score(targetAll, outputAll)
        print("!!!!!!!!!!!!!!!!!!!current band combination: ", accuracy)
        
        if id ==0:
            probsAllAll=probsAll
        else:
            probsAllAll=probsAll+probsAllAll
        id=id+1
        
        outputAll = np.uint8(np.argmax(probsAllAll, axis=1))
        # !!!!!!!!!!
        accuracy = accuracy_score(targetAll, outputAll)
        print("!!!!!!!!!!!!!!!!!!!voted: ", accuracy)
        
    c_report = classification_report(targetAll, outputAll)
    print(c_report)


    # save the results 
    List=[dataset, args.model_name, shot, accuracy]
    # Open our existing CSV file in append mode
    # Create a file object for this file
    with open(os.path.join(root, 'acc_github.csv'), 'a') as f_object:
    
        # Pass this file object to csv.writer()
        # and get a writer object
        writer_object = writer(f_object)
    
        # Pass the list as an argument into
        # the writerow()
        writer_object.writerow(List)
    
        #Close the file object
        f_object.close()


if __name__ == '__main__':

    args, config = get_args_parser()
    main(args, config)