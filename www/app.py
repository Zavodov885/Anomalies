from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

# Обработчик загрузки файлов
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    df = None
    
    try:
        if file.filename.endswith('.json'):
            df = pd.read_json(file)
        elif file.filename.endswith('.xls') or file.filename.endswith('.xlsx'):
            df = pd.read_excel(file)
        
        if df is not None:
            result = process_data(df)
            return jsonify(result)
        else:
            return 'Ошибка чтения файла'
    except Exception as e:
        return str(e)

# Функция обработки данных
def process_data(df):
    results = {}
    columns = list(df.columns)
    
    for col in columns:
        column_data = df[col].values.tolist()
        
        # Методы обнаружения аномалий
        z_score_results = detect_anomalies_z_score(column_data)
        dbscan_results = detect_anomalies_dbscan(column_data)
        
        results[col] = {
            'Z-SCORE': z_score_results,
            'DBSCAN': dbscan_results
        }
    
    return results

# Функция для обработки данных
def process_data(df):
    results = {}
    columns = list(df.select_dtypes(include='number').columns)  # Берём только числовые колонки
    
    
    for col in columns:
        column_data = df[col].dropna().values.tolist()  # Удаляем NaN-значения
        
        if len(column_data) >= 1:  # Проверяем количество элементов
            z_score_results = detect_anomalies_z_score(column_data)
            dbscan_results = detect_anomalies_dbscan(column_data)
            
            results[col] = {
                'Z-SCORE': z_score_results,
                'DBSCAN': dbscan_results
            }
        else:
            print(f"В колонке '{col}' недостаточно данных.")
    
    return results

# Функции для детектирования аномалий
def detect_anomalies_z_score(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    z_scores = [(value - mean) / std_dev for value in data]
    threshold = 3
    anomalies = []
    for i, score in enumerate(z_scores):
        if abs(score) > threshold:
            anomalies.append((i+1, f'Значение {data[i]} является аномалией'))
    return anomalies

def detect_anomalies_dbscan(data):
    if len(data) < 1:
        raise ValueError("Недостаточно данных для применения DBSCAN")
    
    dbscan = DBSCAN(eps=0.5, min_samples=2)
    labels = dbscan.fit_predict(np.array(data).reshape(-1, 1))
    anomalies = []
    for idx, label in enumerate(labels):
        if label == -1:
            anomalies.append((idx + 1, f'Значение {data[idx]} является аномалией'))
    return anomalies

# Вспомогательные функции для методов обнаружения аномалий
def detect_anomalies_z_score(data):
    mean = np.mean(data)
    std_dev = np.std(data)  
    z_scores = [(value - mean) / std_dev for value in data]
    threshold = 3
    anomalies = []
    for i, score in enumerate(z_scores):
        if abs(score) > threshold:
            anomalies.append((i+1, f'Значение {data[i]} является аномалией'))
    return anomalies

def detect_anomalies_dbscan(data):
    dbscan = DBSCAN(eps=0.5, min_samples=2)
    labels = dbscan.fit_predict(np.array(data).reshape(-1, 1))
    anomalies = []
    for idx, label in enumerate(labels):
        if label == -1:
            anomalies.append((idx + 1, f'Значение {data[idx]} является аномалией'))
    return anomalies

if __name__ == '__main__':
    app.run(debug=True)