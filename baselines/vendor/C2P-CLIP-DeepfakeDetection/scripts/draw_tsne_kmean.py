import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
from random import shuffle
from matplotlib import pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
from utils.logger import Progbar
import torch.nn as nn
import torch
from torch.nn import functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import ImageFile
import skimage.io as io
import PIL.Image
import torchvision.datasets as datasets
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
np.random.seed(123)
from kmeans_pytorch import kmeans, kmeans_predict
# from clipcap import gg_text, get_model
from decode_clipfeature_image import get_clipcap_model, get_text, get_clip_model

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

class dataset_folder(datasets.DatasetFolder):
    def __init__( self, root: str, transform: Optional[Callable] = None ):
        super().__init__(root, transform=None, extensions=IMG_EXTENSIONS, loader=None)
        self.processor    = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        image   = PIL.Image.fromarray( io.imread(path) )#.to(device)
        sample  = self.processor(images=image, return_tensors="pt")['pixel_values'][0]
        return sample, target, path

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def binary_dataset(root):
    return dataset_folder(root)

def generate_colors(num_colors):
    # cmap = plt.cm.get_cmap('tab20')
    cmap = plt.colormaps.get_cmap('tab20')
    colors = cmap(range(num_colors))
    return colors
    
def tsne_vis(features, labels, draw_dir, opt, device):
    
    clipcap_model, tokenizer = get_clipcap_model(model_path='https://www.now61.com/f/Xljmi0/coco_prefix_latest.pt', device=device)
    all_labels = list( set( labels.reshape([-1]) ) )
    num_clusters = 3    
    all_cluster_centers = []
    all_text = {}
    for tmp_label in all_labels:
        feature_ = features[labels==tmp_label]
        cluster_ids_x, cluster_centers = kmeans(X=torch.from_numpy(feature_).cuda(), num_clusters=num_clusters, distance='euclidean', device=device)
        all_cluster_centers.append(cluster_centers)
        print('='*100)
        print(opt.legend[tmp_label])
        tmp_text=[]
        for index in range(num_clusters):
            tmp_text.append( get_text(cluster_centers[index].to(device), tokenizer, clipcap_model, fc_path='https://www.now61.com/f/qwvoH5/fc_parameters.pth', cal_detection_feat=False, device=device) )
            print(f'\n{tmp_text[-1]}')
        all_text[str(tmp_label+len(all_labels))] = tmp_text
        print('='*100)

    all_cluster_centers = torch.cat(all_cluster_centers, 0).cpu().numpy()
    features = np.concatenate((features, all_cluster_centers), axis=0)
    new_label = []
    for index, all_label in enumerate(all_labels):
        new_label.extend([len(all_labels)+index]*num_clusters)
    labels = np.concatenate((labels, np.array(new_label)), axis=0)

    embedding_path = os.path.join(draw_dir, '{}_embedding.npy'.format(opt.save_name))

    if opt.do_fit or not os.path.exists(embedding_path):
        print(f">>> t-SNE fitting")
        tsne_model = TSNE(n_jobs=64, perplexity=opt.perplexity, random_state=1024, learning_rate=1000)
        embeddings = tsne_model.fit_transform(features)
        print(f"<<< fitting over")
        np.save(embedding_path, embeddings)        
    else:
        embeddings=np.load(embedding_path)

    index = [i for i in range(len(embeddings))]
    shuffle(index)
    embeddings = [embeddings[index[i]] for i in range(len(index))]
    labels = [labels[index[i]] for i in range(len(index))]
    embeddings = np.array(embeddings)

    print(f">>> draw image")
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]
    plt.figure(figsize=(35,35))
    plt.rcParams['figure.dpi'] = 1000
    colors= generate_colors(20)
    num_classes = len(set(labels))
    
    for i in range(num_classes):
        s = 1000 if i>5 else 20
        marker = '*' if i>5 else 'o'
        color = colors[i]
        class_index = [j for j,v in enumerate(labels) if v == i]
        plt.scatter(vis_x[class_index], vis_y[class_index], s = s, color=color, marker = marker)
        
        
        if i>5 and opt.draw_text:
            texthighs = [0]*len(vis_y[class_index])
            sorted_A = sorted(enumerate(vis_y[class_index]), key=lambda x: x[1], reverse=False)
            for rank, (index, value) in enumerate(sorted_A):
                texthighs[index] = (rank + 1)*40
            for tx, ty, ttext, texthigh in zip(vis_x[class_index], vis_y[class_index], all_text[str(i)], texthighs):
                fc=colors[i-num_classes//2]
                plt.annotate(f' {ttext} ', 
                             xy=(tx, ty), 
                            xycoords="data",
                            xytext=(100, texthigh),
                            fontsize = 30,
                            textcoords="offset points",
                            color="white",
                            va="center",
                            ha="center",
                            weight="bold",
                            bbox=dict(boxstyle="round", fc=fc, ec="none"),
                            arrowprops=dict(
                                connectionstyle="angle3,angleA=0,angleB=90",
                                arrowstyle="wedge, tail_width=1.", 
                                fc=fc, ec="none", patchA=None
                            )
                            )
    if opt.draw_text: img_path = os.path.join(draw_dir,  '{}_draw-text_tsne.png'.format(opt.save_name))
    else:             img_path = os.path.join(draw_dir,  '{}_tsne.png'.format(opt.save_name))
    plt.xticks([])
    plt.yticks([])
    legend = plt.legend(opt.legend, prop = {'size':35})
    for handle in legend.legend_handles:
        handle.set_sizes([300]) 
    plt.show()
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0.1)
    print(f"<<<save image")


def extract_feature(model, draw_loader, device):

    fc_path = 'https://www.now61.com/f/qwvoH5/fc_parameters.pth'
    mod = torch.hub.load_state_dict_from_url( fc_path, map_location="cpu", progress=True ) if fc_path.startswith("http") else torch.load(fc_path, map_location="cpu")
    weight, bias =  mod['fc.weight'].to(device), mod['fc.bias'].to(device)

    features = None
    model.eval()
    progbar = Progbar(len(draw_loader), stateful_metrics=['run-type'])
    with torch.no_grad():
        for _, batch in enumerate(draw_loader):
            input_img_batch, label_batch, path_batch = batch 
            input_img = input_img_batch.to(device)
            label = label_batch.reshape((-1)).to(device)

            feature = model.get_image_features(input_img)
            feature = torch.mul(feature, weight) + bias
            feature /= feature.norm(2, dim=-1, keepdim=True)
            if features is None:
                features=feature.cpu().numpy()
                gt_labels = label
            else:
                gt_labels = torch.cat([gt_labels, label])
                features=np.vstack((features, feature.cpu().numpy()))
                
            progbar.add(1, values=[('run-type', 'extract feature')])
    
    gt_labels = gt_labels.cpu().numpy()
    
    return features, gt_labels

def parse_args():
    parser = argparse.ArgumentParser(description='draw tsne')
    parser.add_argument('--draw_data_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, help='image_path', default='')
    parser.add_argument('--device', default='cuda:0', type=str, help='cuda:n or cpu')
    parser.add_argument('--do_extract', action='store_true', default=False, help='whether to extract features')
    parser.add_argument('--do_fit', action='store_true', default=False, help='whether to fit tsne model')
    parser.add_argument('--save_name', default='cross_all', type=str)
    parser.add_argument('--legend', nargs='+', help='legend')
    parser.add_argument('--draw_text', default=0, type=int)
    parser.add_argument('--perplexity',default=20,type=int)
    args = parser.parse_args()
    def print_options(parser, args):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            default = parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
    print_options(parser, args)
    return args


if __name__ == '__main__':
    opt = parse_args()
    device = torch.device(opt.device)

    draw_dir = os.path.join(os.path.splitext(opt.draw_data_path)[0], "tsne-"+opt.save_name)
    os.makedirs(draw_dir, exist_ok=True)
    feature_path = os.path.join(draw_dir,  '{}_features.npy'.format(opt.save_name))
    label_path = os.path.join(draw_dir,  '{}_labels.npy'.format(opt.save_name))
    print('draw dir: %s' % draw_dir)

    clipmodel, processor = get_clip_model(clip_name='openai/clip-vit-large-patch14', device = device)

    draw_loader = DataLoader(
        dataset=binary_dataset(opt.image_path),
        num_workers=8,
        batch_size=1,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn
    )

    if opt.do_extract or not os.path.exists(feature_path):
        features, gt_labels = extract_feature(clipmodel, draw_loader, device)
        np.save(feature_path, features)
        np.save(label_path, gt_labels)
    else:
        features = np.load(feature_path)
        gt_labels = np.load(label_path)
        
    print('labels:', gt_labels.shape, 'features:', features.shape)
    tsne_vis(features, gt_labels, draw_dir, opt, device)

# CUDA_VISIBLE_DEVICES=1 python draw_tsne_kmean.py --draw_data_path A_tsne_png_20240812  --image_path ../stylegan_tsne_data  --save_name stylegan_test   --legend stylegan-bedroom-real  stylegan-bedroom-fake  stylegan-car-real  stylegan-car-fake  stylegan-cat-real  stylegan-cat-fake  --do_extract --do_fit --draw_text 0