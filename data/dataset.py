import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
import albumentations as A # 데이터 증강용으로 쓰는 라이브러리. 학습용 데이터가 numpy 형식으로 받아서 이걸 사용함. 
from albumentations.pytorch import ToTensorV2 # 텐서 데이터화
# transform으로 증강하고자 한다면 PIL.Image 형식으로 받아도 됨.

class QuickDrawDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # NumPy 배열
        label = self.labels[idx]

        # NumPy 배열을 PIL 이미지로 변환 (흑백 이미지이므로 'L' 모드 사용)
        image = Image.fromarray(image.astype('uint8'), mode='L')
        image = ImageOps.invert(image)  # 흑백 이미지 반전. 학습시키는 그림과 실제 그리는 그림 색상이 반대여서 이렇게 조정

        # Transform 적용
        if self.transform:
            #image = self.transform(image=np.array(image))['image'] # albumentations 결과는 딕셔너리 형식이라 이렇게 전달해야함
            augmented = self.transform(image=np.array(image))
            image = augmented['image']  # 알버멘테이션 결과는 딕셔너리에서 'image' 키로 꺼내야 함
        else:
            # 기본 변환 (텐서 변환 + 정규화)
            image = transforms.ToTensor()(image)

        return image, label

def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
    """Create PyTorch DataLoaders for train, validation, and test sets"""
    # transform 적용. 여기서 데이터 증강 좀 해야하는데.
    # transform = transforms.ToTensor() # 기본 transform
    #transform = transforms.Compose([
    #    # transforms.Resize((64, 64)), # 이미 사이즈가 64x64 라서 주석화
    #    transforms.ToTensor(),
    #    transforms.Normalize((0.5,), (0.5,))  # -1 ~ 1로 정규화
    #])
    
    # 학습 데이터용 데이터 증강 적용
    train_transform = A.Compose([
        #A.RandomCrop(width=56, height=56, p=1.0),  # 랜덤 크롭
        A.Resize(64, 64),  # 크기 복원
        A.HorizontalFlip(p=0.5),  # 수평 대칭
        A.VerticalFlip(p=0.5),  # 수직 대칭
        A.Rotate(limit=90, p=1.0),  # 360도 회전
        A.Affine(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.7),  # 이동, 확대/축소, 회전
        A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0, p=1.0),  # 정규화
        ToTensorV2(),  # Tensor로 변환
    ])

    # 검증 및 테스트 데이터는 증강 없이 원본만 사용
    val_test_transform = A.Compose([
        A.Resize(64, 64),  # 크기만 조정 (그 외 변형 없음)
        A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0, p=1.0),  # 정규화
        ToTensorV2(),  # Tensor로 변환
    ])
    
    # 데이터셋 생성
    train_dataset = QuickDrawDataset(X_train, y_train, transform=train_transform)
    val_dataset = QuickDrawDataset(X_val, y_val, transform=val_test_transform)
    test_dataset = QuickDrawDataset(X_test, y_test, transform=val_test_transform)
    
    # 데이터 불러오는 값들 생성
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader