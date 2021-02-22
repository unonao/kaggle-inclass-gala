
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

import os
""" jupyter のみ
if not os.path.exists('/home/gala/input/image-fmix/FMix-master/fmix.zip'):
    os.makedirs("/home/gala/input/image-fmix/FMix-master/", exist_ok=True)
    os.chdir('/home/gala/input/image-fmix/FMix-master')
    !wget -O fmix.zip https://github.com/ecs-vlc/fmix/archive/master.zip
    !unzip -qq fmix.zip
    !mv FMix-master/* ./
    !rm -r FMix-master
    os.chdir('/home/gala/working')
"""
package_paths = [
    '../input/image-fmix/FMix-master'
]
import sys
for pth in package_paths:
    sys.path.append(pth)

from fmix import sample_mask, make_low_freq_image, binarise_mask


from .utils import get_img, rand_bbox

class GalaDataset(Dataset):
    def __init__(self, df, data_root,
                 shape, # 追加
                 transforms=None,
                 output_label=True,
                 one_hot_label=False,
                 do_fmix=False,
                 fmix_params={
                     'alpha': 1.,
                     'decay_power': 3.,
                     'max_soft': True,
                     'reformulate': False
                 },
                 do_cutmix=False,
                 cutmix_params={
                     'alpha': 1,
                 },
                 image_name_col = "Image",
                 label_col = "label"
                ):

        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.do_fmix = do_fmix
        self.fmix_params = fmix_params
        self.do_cutmix = do_cutmix
        self.cutmix_params = cutmix_params

        self.output_label = output_label
        self.one_hot_label = one_hot_label
        self.image_name_col = image_name_col
        self.label_col = label_col

        if output_label == True:
            self.labels = self.df[self.label_col].values
            #print(self.labels)

            if one_hot_label is True:
                self.labels = np.eye(self.df[self.label_col].max()+1)[self.labels]
                #print(self.labels)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):

        # get labels
        if self.output_label:
            target = self.labels[index]

        img  = get_img("{}/{}".format(self.data_root, self.df.loc[index][self.image_name_col]))

        if self.transforms:
            img = self.transforms(image=img)['image']

        if self.do_fmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            with torch.no_grad():
                #lam, mask = sample_mask(**self.fmix_params)

                lam = np.clip(np.random.beta(self.fmix_params['alpha'], self.fmix_params['alpha']),0.6,0.7)

                # Make mask, get mean / std
                mask = make_low_freq_image(self.fmix_params['decay_power'], shape)
                mask = binarise_mask(mask, lam, shape, self.fmix_params['max_soft'])

                fmix_ix = np.random.choice(self.df.index, size=1)[0]
                fmix_img  = get_img("{}/{}".format(self.data_root, self.df.iloc[fmix_ix][self.image_name_col]))

                if self.transforms:
                    fmix_img = self.transforms(image=fmix_img)['image']

                mask_torch = torch.from_numpy(mask)

                # mix image
                img = mask_torch*img+(1.-mask_torch)*fmix_img

                # mix target
                rate = mask.sum()/shape[0]/shape[1]
                target = rate*target + (1.-rate)*self.labels[fmix_ix]


        if self.do_cutmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            #print(img.sum(), img.shape)
            with torch.no_grad():
                cmix_ix = np.random.choice(self.df.index, size=1)[0]
                cmix_img  = get_img("{}/{}".format(self.data_root, self.df.iloc[cmix_ix]['image_id']))
                if self.transforms:
                    cmix_img = self.transforms(image=cmix_img)['image']

                lam = np.clip(np.random.beta(self.cutmix_params['alpha'], self.cutmix_params['alpha']),0.3,0.4)
                bbx1, bby1, bbx2, bby2 = rand_bbox(shape, lam)

                img[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2, bby1:bby2]

                rate = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (shape[0]*shape[1]))
                target = rate*target + (1.-rate)*self.labels[cmix_ix]

        if self.output_label == True:
            return img, target
        else:
            return img
