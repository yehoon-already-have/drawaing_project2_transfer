import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler  # ✅ Mixed Precision 추가

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    """Train PyTorch model"""
    model = model.to(device)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)  # ✅ 스케줄러 활성화

    scaler = GradScaler()  # ✅ Mixed Precision을 위한 GradScaler 추가

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, labels = inputs.to(device).float(), labels.to(device).long()
            optimizer.zero_grad()

            with autocast():  # ✅ Mixed Precision 연산 적용
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()  # ✅ Scaler 적용
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += torch.count_nonzero(predicted == labels).item()  # ✅ 벡터 연산으로 정확도 최적화

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs, labels = inputs.to(device).float(), labels.to(device).long()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += torch.count_nonzero(predicted == labels).item()  # ✅ 벡터 연산으로 정확도 최적화

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total

        scheduler.step(val_loss)  # ✅ 학습률 감소 적용

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    return model, history

def evaluate_model(model, test_loader, criterion, device='cuda'):
    """Evaluate model performance"""
    model = model.to(device)
    model.eval()

    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device).float(), labels.to(device).long()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += torch.count_nonzero(predicted == labels).item()  # ✅ 벡터 연산으로 정확도 최적화

    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_correct / test_total

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    return test_loss, test_acc

def get_predictions(model, data_loader, device='cuda', top_k=1):  # ✅ top_k 추가
    """Get model top-k predictions for a data loader"""
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Getting predictions"):
            inputs = inputs.to(device).float()
            
            outputs = model(inputs)
            top_probs, top_classes = torch.topk(outputs, k=top_k, dim=1)  # ✅ top-k 적용
            
            all_predictions.extend(top_classes.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return all_predictions, all_labels