import argparse
import os
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from torch.utils.data import Dataset
from models import get_model
from PIL import Image
import pickle
from tqdm import tqdm
from io import BytesIO
from dataset_paths import DATASET_PATHS
import random
import shutil
from scipy.ndimage.filters import gaussian_filter

from metric_utils import evaluate_metrics
from lmdb_utils import LMDBReader, ensure_pil

SEED = 0
def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}





def _gather_tensor(tensor, world_size, pad_value=-1):
    local_len = torch.tensor([tensor.numel()], device=tensor.device, dtype=torch.long)
    gathered_lens = [torch.zeros_like(local_len) for _ in range(world_size)]
    dist.all_gather(gathered_lens, local_len)
    max_len = int(torch.stack(gathered_lens).max().item())

    if tensor.numel() < max_len:
        pad_size = max_len - tensor.numel()
        tensor = torch.cat([tensor, torch.full((pad_size,), pad_value, device=tensor.device)])

    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    merged = torch.cat(gathered)
    return merged
        

 
def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality) # ranging from 0-95, 75 is default
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return Image.fromarray(img)


def gaussian_blur(img, sigma):
    img = np.array(img)

    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

    return Image.fromarray(img)



def validate(model, loader, device=None, distributed=False, world_size=1, rank=0, desc="Validating"):
    if loader is None:
        return None, None, None

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        y_true, y_pred = [], []
        show_progress = (not distributed) or rank == 0
        progress_bar = tqdm(total=len(loader), desc=desc, dynamic_ncols=True) if show_progress else None
        for img, label in loader:
            in_tens = img.to(device, non_blocking=True)
            preds = model(in_tens).sigmoid().flatten()
            y_pred.extend(preds.detach().cpu().tolist())
            y_true.extend(label.flatten().tolist())

            if progress_bar is not None:
                progress_bar.update(1)

        if progress_bar is not None:
            progress_bar.close()

    if distributed:
        true_tensor = torch.tensor(y_true, device=device, dtype=torch.float32)
        pred_tensor = torch.tensor(y_pred, device=device, dtype=torch.float32)

        true_tensor = _gather_tensor(true_tensor, world_size)
        pred_tensor = _gather_tensor(pred_tensor, world_size)

        mask = true_tensor >= 0
        y_true = true_tensor[mask].cpu().numpy()
        y_pred = pred_tensor[mask].cpu().numpy()
    else:
        y_true, y_pred = np.array(y_true), np.array(y_pred)

    metrics = evaluate_metrics(y_true, y_pred)
    return metrics, y_true, y_pred

    
    



# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 




def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg", "bmp"]):
    out = [] 
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts)  and  (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def get_list(path, must_contain=''):
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [ item for item in image_list if must_contain in item   ]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list





class RealFakeDataset(Dataset):
    def __init__(self,  real_path, 
                        fake_path, 
                        data_mode, 
                        max_sample,
                        arch,
                        jpeg_quality=None,
                        gaussian_sigma=None):

        assert data_mode in ["wang2020", "ours"]
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma
        
        # = = = = = = data path = = = = = = = = = # 
        if type(real_path) == str and type(fake_path) == str:
            real_list, fake_list = self.read_path(real_path, fake_path, data_mode, max_sample)
        else:
            real_list = []
            fake_list = []
            for real_p, fake_p in zip(real_path, fake_path):
                real_l, fake_l = self.read_path(real_p, fake_p, data_mode, max_sample)
                real_list += real_l
                fake_list += fake_l

        self.total_list = real_list + fake_list


        # = = = = = =  label = = = = = = = = = # 

        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1

        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
        ])


    def read_path(self, real_path, fake_path, data_mode, max_sample):

        if data_mode == 'wang2020':
            real_list = get_list(real_path, must_contain='0_real')
            fake_list = get_list(fake_path, must_contain='1_fake')
        else:
            real_list = get_list(real_path)
            fake_list = get_list(fake_path)


        if max_sample is not None:
            if (max_sample > len(real_list)) or (max_sample > len(fake_list)):
                max_sample = 100
                print("not enough images, max_sample falling to 100")
            random.shuffle(real_list)
            random.shuffle(fake_list)
            real_list = real_list[0:max_sample]
            fake_list = fake_list[0:max_sample]

        assert len(real_list) == len(fake_list)  

        return real_list, fake_list



    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        
        img_path = self.total_list[idx]

        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")

        if self.gaussian_sigma is not None:
            img = gaussian_blur(img, self.gaussian_sigma) 
        if self.jpeg_quality is not None:
            img = png2jpg(img, self.jpeg_quality)

        img = self.transform(img)
        return img, label


class LMDBEvalDataset(Dataset):
    def __init__(self, lmdb_path, arch):
        self.reader = LMDBReader(lmdb_path, decode=True, mode="PIL")
        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
        ])

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, index):
        image, label = self.reader[index]
        if isinstance(image, list):
            image = image[0]
        if isinstance(label, dict):
            label = int(label.get('label', label.get('binary_label', 0)))
        else:
            label = int(label)
        image = ensure_pil(image).convert('RGB')
        image = self.transform(image)
        return image, torch.tensor(label)





if __name__ == '__main__':


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--real_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--fake_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--lmdb_path', type=str, default=None, help='optional lmdb path for evaluation')
    parser.add_argument('--data_mode', type=str, default=None, help='wang2020 or ours')
    parser.add_argument('--max_sample', type=int, default=1000, help='only check this number of images for both fake/real')

    parser.add_argument('--arch', type=str, default='res50')
    parser.add_argument('--ckpt', type=str, default='./pretrained_weights/fc_weights.pth')

    parser.add_argument('--result_folder', type=str, default='result', help='')
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--jpeg_quality', type=int, default=None, help="100, 90, 80, ... 30. Used to test robustness of our model. Not apply if None")
    parser.add_argument('--gaussian_sigma', type=int, default=None, help="0,1,2,3,4.     Used to test robustness of our model. Not apply if None")


    opt = parser.parse_args()

    
    if os.path.exists(opt.result_folder):
        shutil.rmtree(opt.result_folder)
    os.makedirs(opt.result_folder, exist_ok=True)

    model = get_model(opt.arch)
    state_dict = torch.load(opt.ckpt, map_location='cpu')
    model.fc.load_state_dict(state_dict)
    print ("Model loaded..")
    model.eval()
    model.cuda()

    if opt.lmdb_path:
        dataset_paths = [dict(key=os.path.basename(opt.lmdb_path), lmdb_path=opt.lmdb_path)]
    elif (opt.real_path is None) or (opt.fake_path is None) or (opt.data_mode is None):
        dataset_paths = DATASET_PATHS
    else:
        dataset_paths = [dict(real_path=opt.real_path, fake_path=opt.fake_path, data_mode=opt.data_mode)]

    for dataset_path in dataset_paths:
        set_seed()

        if dataset_path.get('lmdb_path'):
            dataset = LMDBEvalDataset(dataset_path['lmdb_path'], opt.arch)
        else:
            dataset = RealFakeDataset(
                dataset_path['real_path'],
                dataset_path['fake_path'],
                dataset_path['data_mode'],
                opt.max_sample,
                opt.arch,
                jpeg_quality=opt.jpeg_quality,
                gaussian_sigma=opt.gaussian_sigma,
            )

        loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)
        metrics, _, _ = validate(model, loader, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                 distributed=False, world_size=1, rank=0,
                                 desc=f"Eval {dataset_path.get('key', dataset_path.get('lmdb_path', 'dataset'))}")

        with open(os.path.join(opt.result_folder, 'metrics.txt'), 'a') as f:
            f.write(
                f"{dataset_path.get('key', dataset_path.get('lmdb_path', 'dataset'))}: "
                f"acc={metrics['acc']*100:.2f}; pr_auc={metrics['pr_auc']*100:.2f}; "
                f"auroc={metrics['auroc']*100:.2f}; avg_recall={metrics['avg_recall']*100:.2f}; "
                f"tpr@0.05fpr={metrics['tpr_at_fpr']*100:.2f}\n"
            )
