# predict.py
import argparse
from models_services.classifier import ImageClassifier
import torch

def classifier_predict():
    parser = argparse.ArgumentParser(description='圖像分類預測')
    parser.add_argument('--model', type=str, required=True, help='模型路徑')
    parser.add_argument('--image', type=str, required=True, help='要預測的圖片路徑')
    parser.add_argument('--class-names', type=str, nargs='+', required=True, help='類別名稱列表')
    
    args = parser.parse_args()
    
    # 加載模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    classifier = ImageClassifier.load_model(args.model, device)
    classifier.class_names = args.class_names  # 更新類別名稱
    
    # 進行預測
    result = classifier.predict(args.image)
    
    print("\n預測結果:")
    print(f"最可能的類別: {result['class_name']}")
    print(f"置信度: {result['confidence']*100:.2f}%")
    print("\n所有類別機率:")
    for class_name, prob in zip(result['class_names'], result['probabilities']):
        print(f"{class_name}: {prob*100:.2f}%")