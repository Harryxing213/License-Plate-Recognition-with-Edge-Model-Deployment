from torch.utils.data import Dataset
import torch
import cv2
from pathlib import Path

VOCAB = ['<PAD>', '<BOS>', '<EOS>', '皖', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑', '苏', '浙', '京', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤', '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新', '警', '学', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

class PlateDataset(Dataset):
    def __init__(self,src_path):
        super().__init__()

        self.src_path = Path(src_path)
        self.collect = self.data_collect()

    def data_collect(self):
        src_path = self.src_path
        label_list = src_path / "label.txt"
        collect = []

        with open(label_list, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                img_name, label = line.split("\t")
                img_path = str(src_path / "images" / img_name)
                collect.append([img_path, label])
        if not collect:
            raise ValueError(f"No labels found: {label_list}")
        return collect


    def prepare_label(self, label):
        if len(label) != 7:
            raise ValueError(f"The final model expects a 7-character plate: {label}")
        map = VOCAB
        idx_code = [1,]
        idx_code.append(map.index(label[0]))
        idx_code.append(map.index(label[1]))
        for ch in label[2:]:
            idx_code.append(map.index(ch))
        idx_code.append(2)
        decoder_input = idx_code[:-1]
        target = idx_code[1:]
        # print(f"{label}\n",idx_code)

        return idx_code, decoder_input, target

    @staticmethod
    def image_process(img_path, width=256, height=64):
        '''desired shape [3, 64, 256]
           dtype: float32
           value range: [0,1]'''
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = cv2.resize(img, (width, height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        img = img.transpose(2, 0, 1)
        return img

    def __getitem__(self, index):
        img_path = self.collect[index][0]
        label = self.collect[index][1]
        _, decoder_input, target = self.prepare_label(label)
        img = self.image_process(img_path, width=256, height=64)
        img = torch.tensor(img, dtype=torch.float32)
        decoder_input = torch.tensor(decoder_input, dtype=torch.int64)
        target = torch.tensor(target, dtype=torch.int64)

        return img, decoder_input, target

    def __len__(self):
        return len(self.collect)
