from flask import Flask, render_template, request, jsonify
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import base64
from io import BytesIO

app = Flask(__name__)

# Списки для проверки фруктов и овощей
FRUITS = {"apple", "banana", "pear", "orange", "kiwi", "grape", "strawberry", "blueberry"}
VEGETABLES = {"carrot", "potato", "tomato", "cucumber", "lettuce", "broccoli", "carrot"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'json_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['json_file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        data = json.load(file)
       
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                df = pd.DataFrame(data)
            else:

                df = pd.DataFrame({'value': data})
        else:
            df = pd.DataFrame([data])


        anomalies_stat = detect_anomalies_statistical(df)
        anomalies_graph = detect_anomalies_graph(df)

        # Создание таблицы с выделением аномалий
        table_html = create_table_with_anomalies(df, anomalies_stat, anomalies_graph)

        # Создание графика
        plot_url = create_plot(df, anomalies_stat, anomalies_graph)


        anomalies_values = {}
        all_anomalies = [('stat', anomalies_stat), ('graph', anomalies_graph)]
        for method_name, anoms in all_anomalies:
            anomalies_values[method_name] = {}
            for col, indices in anoms.items():
                anomalies_values[method_name][col] = df.loc[indices, col].tolist()

        return jsonify({
            'table': table_html,
            'graph': plot_url,
            'anomalies_values': anomalies_values
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def detect_anomalies_statistical(df):
    anomalies = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        z_scores = np.abs(stats.zscore(df[col]))
        anomalies[col] = df[z_scores > 3].index.tolist()
    for col in df.select_dtypes(include=['object']).columns:

        unique_values = set(df[col].dropna().astype(str).str.lower())
        fruits_in = unique_values & FRUITS
        veggies_in = unique_values & VEGETABLES
        if len(fruits_in) > len(veggies_in) and len(veggies_in) > 0:

            anomalies[col] = df[df[col].astype(str).str.lower().isin(veggies_in)].index.tolist()
        elif len(veggies_in) > len(fruits_in) and len(fruits_in) > 0:

            anomalies[col] = df[df[col].astype(str).str.lower().isin(fruits_in)].index.tolist()
        else:

            value_counts = df[col].value_counts()
            total = len(df)
            rare_values = value_counts[value_counts <= max(1, total * 0.05)].index
            if len(rare_values) < total:
                anomalies[col] = df[df[col].isin(rare_values)].index.tolist()
            else:
                anomalies[col] = []
    return anomalies

def detect_anomalies_clustering(df):
    anomalies = {}

    for col in df.select_dtypes(include=[np.number]).columns:
        if len(df) > 1:
            n_clusters = min(3, len(df))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(df[[col]])
            distances = kmeans.transform(df[[col]]).min(axis=1)
            threshold = np.percentile(distances, 95)
            anomalies[col] = df[distances > threshold].index.tolist()
        else:
            anomalies[col] = []
    return anomalies

def detect_anomalies_graph(df):
    anomalies = {}

    if 'value' in df.columns and len(df.columns) == 1:
        values = df['value'].tolist()

        types = [type(v).__name__ for v in values]
        most_common_type = max(set(types), key=types.count)

        anomaly_indices = [i for i, t in enumerate(types) if t != most_common_type]
        anomalies['value'] = anomaly_indices
    else:

        anomalies = {col: [] for col in df.columns}
    return anomalies

def create_plot(df, anomalies_stat, anomalies_graph):
    plt.figure(figsize=(10, 6))


    anom_indices = set()
    for col_anoms in anomalies_stat.values():
        anom_indices.update(col_anoms)
    if 'graph' in anomalies_graph:
        anom_indices.update(anomalies_graph['graph'])


    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        col = numeric_cols[0]
        plt.hist(df[col], bins=20, alpha=0.7, label='Распределение')

        if anom_indices:
            anom_values = df.loc[list(anom_indices), col]
            plt.scatter(anom_values, [0]*len(anom_values), color='red', s=100, label='Аномалии', zorder=5)

        plt.xlabel('Значение')
        plt.ylabel('Частота')
    else:

        if len(df.columns) == 1:
            col = df.columns[0]
            if col == 'value':
 
                x = df.index
                y_numeric = pd.to_numeric(df[col], errors='coerce')
                y_plot = y_numeric.fillna(0)
                colors = ['red' if i in anom_indices else 'blue' for i in x]
                plt.bar(x, y_plot, alpha=0.7, color=colors, label='Значения')

                plt.xlabel('Индекс')
                plt.ylabel('Значение')
            else:

                value_counts = df[col].value_counts()
                bars = plt.bar(range(len(value_counts)), value_counts.values, alpha=0.7, label='Распределение')
                plt.xticks(range(len(value_counts)), [str(x) for x in value_counts.index], rotation=45)


                for i, val in enumerate(value_counts.index):
                    if df[df[col] == val].index.isin(anom_indices).any():
                        bars[i].set_color('red')

                plt.xlabel('Значение')
                plt.ylabel('Частота')
        else:

            plt.text(0.5, 0.5, 'График для категориальных данных\nс несколькими столбцами не реализован', ha='center', va='center', transform=plt.gca().transAxes)

    plt.title('Distribution plot with anomalies')
    plt.legend()


    plot_path = os.path.join('static', 'plot.png')
    plt.savefig(plot_path)
    plt.close()

    return '/static/plot.png'

def create_table_with_anomalies(df, anomalies_stat, anomalies_graph):

    anomaly_indices = set()
    for col_anoms in anomalies_stat.values():
        anomaly_indices.update(col_anoms)
    if 'graph' in anomalies_graph:
        anomaly_indices.update(anomalies_graph['graph'])


    html = '<table class="table table-striped"><thead><tr>'
    for col in df.columns:
        html += f'<th>{col}</th>'
    html += '</tr></thead><tbody>'

    for idx, row in df.iterrows():
        row_class = 'table-danger' if idx in anomaly_indices else ''
        html += f'<tr class="{row_class}">'
        for val in row:
            html += f'<td>{val}</td>'
        html += '</tr>'

    html += '</tbody></table>'
    return html

if __name__ == '__main__':
    app.run(debug=True)