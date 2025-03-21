# CNN 모델 활용에 쓰일 변수들들

# 데이터 관련련
IMAGE_SIZE = 64
SAMPLES_PER_CATEGORY = 8000
DATA_DIR = 'C:/Users/SSAFY/AppData/Local/Google/Cloud SDK/raw'  
# 위에는 NDJSON 파일 주소

# Training 세팅
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
DEVICE = 'cuda'  # or 'cpu' 

# 저장할 모델을 넣을 경로(디렉토리)
MODEL_SAVE_DIR = 'result'