import os
from torch.utils.data import Dataset
from PIL import Image

class BrainMRIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)  # ['glioma', 'meningioma', 'pituitary', 'notumor']
        self.samples = []  # will hold (image_path, label_index) pairs

        for class_index, class_name in enumerate(self.classes):
            class_folder = os.path.join(self.root_dir, class_name)
            for filename in os.listdir(class_folder):
                image_path = os.path.join(class_folder, filename)
                self.samples.append((image_path, class_index))
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
if __name__ == "__main__":
    dataset = BrainMRIDataset("D:/Downloads/College ZewailCity UST/1- Miscellaneous/4- myProjects/CORTEX (Computational Oncology & Radiological Turmor Examination)/data/Training")
    print(len(dataset))
    print(dataset[0])
