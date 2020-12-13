import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader 
from tqdm import tqdm
from PIL import Image
from argparse import ArgumentParser
from torchvision.models import resnet152
from torchvision.transforms import ToTensor, Normalize


class ImageDataset(Dataset):
    def __init__(self, path_to_images):
        super(ImageDataset, self).__init__()
        self.items = list(map(lambda s: os.path.join(path_to_images, s), os.listdir(path_to_images)))
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    
    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        img_path = self.items[index]
        image = Image.open(img_path)
        image = image.resize((448, 448))
        image = ToTensor()(image)
        # Convert Grayscale image to RGB
        if image.size(0) == 1:
            image = image.repeat(3, 1, 1)
        image = self.normalize(image)
        # image = image.unsqueeze(0)
        npy_fname = img_path.split("/")[-1].split(".")[0] + ".npy"
        return npy_fname, image


def generate_coco_resnet_features_(args):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("using GPU")
    else:
        device = torch.device("cpu")
        print("using CPU")

    feature_generator = resnet152(pretrained=True)
    ch = list(feature_generator.children())
    ch = ch[:-1]
    feature_generator = nn.Sequential(*ch)
    feature_generator.eval().to(device)
    splits = ["train2014", "val2014", "test2015"]

    for split in splits:
        input_dir = os.path.join(args["input_dir"], split)
        output_dir = os.path.join(args["output_dir"], split)
        os.makedirs(output_dir, exist_ok=True)
        dataset = ImageDataset(input_dir)
        dataloader = DataLoader(dataset, batch_size=8)
        file_names = os.listdir(input_dir)
        pbar = tqdm(dataloader)
        pbar.set_description("Generating features of {}".format(split))

        for f_names, images in pbar:
            # file_path = os.path.join(input_dir, file_name)
            images = images.to(device)
            with torch.no_grad():
                features = feature_generator(images)
            features = features.squeeze().tolist()
            for f_name, feat in zip(f_names, features):
                feat = np.array(feat)
                file_npy_path = os.path.join(output_dir, f_name)
                np.save(file_npy_path, feat)

def generate_coco_resnet_features(args):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("using GPU")
    else:
        device = torch.device("cpu")
        print("using CPU")

    feature_generator = resnet152(pretrained=True)
    ch = list(feature_generator.children())
    ch = ch[:-1]
    feature_generator = nn.Sequential(*ch)
    feature_generator.eval().to(device)
    splits = ["train2014", "val2014", "test2015"]
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    for split in splits:
        input_dir = os.path.join(args["input_dir"], split)
        output_dir = os.path.join(args["output_dir"], split)
        os.makedirs(output_dir)
        file_names = os.listdir(input_dir)
        pbar = tqdm(file_names)
        pbar.set_description("Generating features of {}".format(split))
        for file_name in pbar:
            file_path = os.path.join(input_dir, file_name)
            image = Image.open(file_path)
            image = image.resize((448, 448))
            image = ToTensor()(image)
            # Convert Grayscale image to RGB
            if image.size(0) == 1:
                image = image.repeat(3, 1, 1)
            image = normalize(image)
            image = image.unsqueeze(0)
            image = image.to(device)
            with torch.no_grad():
                features = feature_generator(image)
                features = features.squeeze().cpu().numpy()
                file_npy_path = os.path.join(output_dir, file_name.split(".")[0])
                np.save(file_npy_path, features)


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-i", "--input_dir", default="/data/VQA/images/")
    arg_parser.add_argument("-o", "--output_dir", default="/data/VQA/features_2048")
    args = vars(arg_parser.parse_args())
    generate_coco_resnet_features(args)
