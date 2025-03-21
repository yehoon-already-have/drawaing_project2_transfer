import torch
import albumentations as A

print(torch.cuda.is_available())  # GPU가 사용 가능한지 확인
print(torch.cuda.current_device())  # 사용 중인 GPU 디바이스 번호
print(torch.cuda.get_device_name(0))  # GPU 이름 확인




# albumentations에서 사용 가능한 변환 기법들 출력
# print(dir(A))