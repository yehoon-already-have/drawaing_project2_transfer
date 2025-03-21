import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 만든 파이썬 모듈 파일 import
from config import * # 관련 변수들
from data.preprocessing import get_local_categories, process_categories, split_data, normalize_data
from data.dataset import create_data_loaders 
from model.network import DrawAingCNN, save_model, load_model
from model.training import train_model, evaluate_model, get_predictions
from utils.visualization import plot_history, visualize_samples

def main():
    # CUDA 있는지 확인인
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 파일명 읽고 카테고리 생성하기
    print(f"Looking for NDJSON files in {DATA_DIR}...")
    categories = ['ant', 'apple', 'axe', 'backpack', 'banana', 'barn', 'basket', 'bear', 'bed', 'bee', 'bench', 'bread', 'bridge', 'broccoli', 'broom', 'bucket', 'bush', 'butterfly', 'carrot', 'cat', 'chair', 'cloud', 'cow', 'cup', 'dog', 'donut', 'door', 'duck', 'feather', 'fence' ] #, 'fish', 'flower', 'frog', 'garden hose', 'grapes', 'grass', 'hedgehog', 'horse', 'house', 'ladder', 'leaf', 'monkey', 'moon', 'mountain', 'mouse', 'mushroom', 'onion', 'peanut', 'pear', 'pig']
    # get_local_categories(DATA_DIR) 원래는 전체 카테고리를 가져와서 수행했는데, 학습률도 저조하고 속도도 느려서 
    print(f"Found {len(categories)} categories: {categories}")
    
    # 데이터 전처리
    print("Processing drawing data...")
    X, y = process_categories(categories, DATA_DIR, SAMPLES_PER_CATEGORY, IMAGE_SIZE)
    print(f"Processed {len(X)} drawings with shape {X.shape}")
    
    # 데이터 분할
    print("Splitting data into train/val/test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # 데이터 정규화 -> transform할때 이미 정규화를 하므로 생략략
    # X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)
    
    # 데이터 생성
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, BATCH_SIZE)
    
    # 전처리한 데이터 보기 -> 64x64짜리 이미지지
    print("Visualizing sample images...")
    indices = np.random.choice(len(X_train), 10, replace=False)
    visualize_samples(X_train[indices], y_train[indices], categories)
    
    # 모델 초기화(적용용)
    print("Initializing model...")
    model = DrawAingCNN(num_classes=len(categories))
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # 모델의 손실함수와 optimazer 적용. 가중치
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 모델 학습습
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        num_epochs=NUM_EPOCHS, device=device)
    
    # 학습 상태 도식화(학습, 검증증)
    print("Plotting training history...")
    plot_history(history)
    
    # 테스트값 평가가
    print("Evaluating model on test set...")
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    
    # 각 예측 결과에 대한 confusion matrix 생성
    print("Generating classification report and confusion matrix...")
    y_pred, y_true = get_predictions(model, test_loader, device)
    
    # 분류 정확도 표로 표시
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=categories))
    
    # confusion matrix 그래프  
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # 모델 저장 
    print("Saving model...")
    save_model(model, "draw_classify_model", MODEL_SAVE_DIR)
    print(f"Model saved to {MODEL_SAVE_DIR}/draw_classify_model.pth")
    
    print("Done!")

if __name__ == "__main__":
    main()