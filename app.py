import numpy as np
from flask import Flask, request, jsonify, render_template
import json
import pandas as pd
import csv

url = 'https://raw.githubusercontent.com/QuocAnh261100/GQA/master/csvjson_1.csv'
# url = 'https://raw.githubusercontent.com/QuocAnh261100/GQA/master/data_long_lat.csv'
raw_data = pd.read_csv(url)


# with open('test1.json') as f:
#  data = json.load(f)


app = Flask(__name__)


def distance(lat_1, long_1, lat_2, long_2):
    import geopy.distance
    return geopy.distance.great_circle((lat_1, long_1), (lat_2, long_2))


def neighbor_dectection_with_longlat(latitude, longitude, data):
    distances = []
    long_ = longitude
    lat_ = latitude
    for i in range(len(data)):
        distances.append(
            distance(lat_, long_, data.loc[i].latitude, data.loc[i].longitude))
    distances = np.array(distances)
    return data.loc[np.argsort(distances)[:40]]


def neighbor_detection(latitude, longitude, data):
    return neighbor_dectection_with_longlat(latitude, longitude, data)


def percentage_type_house():
    count_type_of_houses = raw_data.loai.value_counts()/len(raw_data) * 100
    dicti = {'Others': np.sum(count_type_of_houses[5:])}
    ti_le = (raw_data.loai.value_counts()/len(raw_data) * 100)
    ti_le_1 = ti_le[:5].append(pd.Series(dicti), ignore_index=False)
    ti_le_1 = pd.DataFrame(ti_le_1).rename(columns={0: 'ti_le'})
    return ti_le_1


def getdatas(data, min_price, max_price, min_area, max_area, housetype):
    data = data[data.loai == housetype].copy()
    return data.loc[(data['giaCa'] > int(min_price)) & (data['giaCa'] < int(max_price)) & (data['dienTich'] < int(max_area)) & (data['dienTich'] > int(min_area))]


def make_json(csvFilePath, jsonFilePath):
    data = {}
    with open(csvFilePath, encoding='utf-8') as csvf:
        csvReader = csv.DictReader(csvf)

        for rows in csvReader:

            key = rows['']
            data[key] = rows

    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json.dumps(data, indent=4))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/getAll')
def getAll():
    raw_data.to_csv("allAddress.csv")
    csvFilePath = r"allAddress.csv"
    jsonFilePath = r'all.json'
    make_json(csvFilePath, jsonFilePath)
    with open('all.json') as f:
        data = json.load(f)
    return data


@app.route('/get40closest/<latitude>/<longitude>')
def get20closest(latitude, longitude):
    neighbor = neighbor_detection(latitude, longitude, raw_data)
    neighbor.to_csv("neighbor.csv")
    csvFilePath = r'neighbor.csv'
    jsonFilePath = r'result.json'
    make_json(csvFilePath, jsonFilePath)
    with open('result.json') as f:
        data = json.load(f)
    return data


@app.route('/getPercentEachType')
def getPercentage():
    percent = percentage_type_house()
    percent.to_csv("percent.csv")
    csvFilePath = r"percent.csv"
    jsonFilePath = r'percent.json'
    make_json(csvFilePath, jsonFilePath)
    with open('percent.json') as f:
        data = json.load(f)
    return data


@app.route('/filter/<min_price>/<max_price>/<min_area>/<max_area>/<housetype>')
def getFilter(min_price, max_price, min_area, max_area, housetype):
    data = pd.read_csv("neighbor.csv")
    result = getdatas(data, min_price, max_price,
                      min_area, max_area, housetype)
    result.to_csv("neighborFilter.csv")
    csvFilePath = r"neighborFilter.csv"
    jsonFilePath = r'filter.json'
    make_json(csvFilePath, jsonFilePath)
    with open('filter.json') as f:
        data = json.load(f)
    return data


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    pass


if __name__ == "__main__":
    app.run(debug=True)
