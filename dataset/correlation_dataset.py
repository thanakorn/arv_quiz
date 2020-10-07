import os.path as osp
import torchvision.transforms as transforms
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset, DataLoader

class CorrelationDataset(Dataset):
    
    def __init__(self, img_root, annotation_file, limit=None):
        super().__init__()
        self.img_dir = img_root
        self.annotation = pd.read_csv(annotation_file)
        # TODO: Remove on the real run
        self.annotation = self.annotation.sort_values(by=['id'])
        self.limit = limit
        
        self.img_transform = transforms.Compose([
            # transforms.Resize((75, 75)),
            transforms.ToTensor()
        ])
        
    def __len__(self):
        if self.limit:
            return min(self.limit, self.annotation.shape[0])
        else:
            return self.annotation.shape[0]
    
    def __getitem__(self, index):
        filename, expected_value = self.annotation.iloc[index]['id'], self.annotation.iloc[index]['corr']
        try:
            img = self.img_transform(Image.open(osp.join(self.img_dir, filename + '.png')))
            return img, expected_value
        except Exception as ex:
            print(f'Unable to load image {ex}')
            raise ex
        