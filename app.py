import numpy as np
from flask import Flask, request, jsonify, render_template
import json
import pandas as pd
import csv
import pickle
import matplotlib.dates as mdates
from datetime import date, timedelta
from fuzzywuzzy import fuzz

# url = 'https://raw.githubusercontent.com/QuocAnh261100/GQA/master/combined_csv.csv'
# url = 'https://raw.githubusercontent.com/QuocAnh261100/GQA/master/data_long_lat.csv'
# raw_data = pd.read_csv(url)

# loaded_pickle = pickle.loads(list_pickle)
# with open('test1.json') as f:
#  data = json.load(f)

# f = open('top10.txt', 'r')
# p = f.read()
# top10 = p.split(' ')
# top10.pop()
# print(top10)

top10c = pd.read_csv('top10.csv')
top10 = top10c['0'].values



# top10 = ['Overall_S', 'dienTich', 'longitude', 'latitude', 'phongTam/dientich','phongngu/dientich', 'ngayDangTin_ts', 'soTang', 'Nhà mặt phố', 'Hệ thống an ninh']
data_cleaned = pd.read_csv('clean.csv')
model = pickle.load(open('pima.pkl', 'rb'))
print(model)
# drop_index = data_cleaned[data_cleaned['diaChi'] == 'Bến Thành, Quận 1, Hà Nội'].index
# data_cleaned = data_cleaned[~drop_index]
# print(drop_index)
app = Flask(__name__)


def distance(lat_1, long_1, lat_2, long_2):
    import geopy.distance
    return geopy.distance.great_circle((lat_1, long_1), (lat_2, long_2))


def neighbor_dectection_with_longlat(latitude, longitude, data):
    distances = []
    long_ = longitude
    lat_ = latitude
    for i in range(len(data)):
        if(data.loc[i].latitude != None and data.loc[i].longitude != None):
            distances.append(
                distance(lat_, long_, data.loc[i].latitude, data.loc[i].longitude))
    distances = np.array(distances)
    return data.loc[np.argsort(distances)[:30]]


def neighbor_detection(latitude, longitude, data):
    return neighbor_dectection_with_longlat(latitude, longitude, data)


def percentage_type_house():
    count_type_of_houses = data_cleaned.loai.value_counts()/len(data_cleaned) * 100
    dicti = {'Khác': np.sum(count_type_of_houses[4:])}
    ti_le = (data_cleaned.loai.value_counts()/len(data_cleaned) * 100)
    ti_le_1 = ti_le[:4].append(pd.Series(dicti), ignore_index=False)
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

# 5. Prediction


def predict_trend_loai(quan, longitude, latitude, model, top10, data_cleaned):
    index_1 = data_cleaned[data_cleaned["tinhThanhPho"].str.split(", ").map(lambda x: len(x)<=1)].index
    data_cleaned.drop(index = index_1, inplace = True)
    data_district = data_cleaned["tinhThanhPho"].str.split(
        ", ").map(lambda x: x[-2])

    highest_match_district = 0
    highest_accuracy = 0
    districts = data_district.unique()
    for i in range(len(districts)):
        acc = fuzz.token_set_ratio(districts[i], quan)
        if (acc > highest_accuracy):
            highest_match_district = i
            highest_accuracy = acc

    quan = districts[highest_match_district]

    house_types = list(data_cleaned[data_district == quan].loai.unique())
    avg_col = []

    for col in data_cleaned.columns:
        if col != 'maTin' and data_cleaned[col].dtypes != 'object' and data_cleaned[col].dtypes != 'datetime64[ns]':
            avg_col.append(col)
    avg_col.append('phapLy')
    avg_col.remove('giaCa')

    house_type_col = list(data_cleaned['loai'].unique())
    # for house_type in house_type_col:
    #   avg_col.remove(house_type)

    bool_col = []
    for col in data_cleaned.columns:
        if data_cleaned[col].dtypes == 'bool' or data_cleaned[col].dtypes == 'int64' and col != 'maTin':
            bool_col.append(col)

    data_new = data_cleaned.copy()
    data_new[bool_col] = data_new[bool_col].astype('float64')
    for col in data_new.columns:
        if col in avg_col:
            data_new[col].fillna(data_new[col].mean(), inplace=True)
    avg_col_dat = list(np.average(data_new[avg_col], 0))

    from datetime import date, timedelta
    import matplotlib.dates as mdates

    import pandas as pd
    data_pred = []
    for house in house_types:
        today = date.today()
        for i in range(5):
            months_added = timedelta(weeks=12 * (i - 3))
            row = []
            row.append(house)
            row.append(today + months_added)
            for dat in avg_col_dat:
                row.append(dat)
            data_pred.append(row)

    lst_of_feature = ['loai', 'ngayDangTin'] + avg_col
    df_for_prediction = pd.DataFrame(data_pred).rename(
        columns=dict(enumerate(lst_of_feature)))

    df_for_prediction['longitude'] = float(longitude)
    df_for_prediction['latitude'] = float(latitude)

    import matplotlib.dates as mdates
    df_for_prediction['ngayDangTin_ts'] = df_for_prediction['ngayDangTin'].apply(
        lambda x: mdates.date2num(x))

    for house_type in house_type_col:
        df_for_prediction[house_type] = 0

    for house_type in house_types:
        idx = df_for_prediction[df_for_prediction['loai'] == house_type].index
        df_for_prediction.loc[idx, house_type] = 1

    # create new features
    cols = df_for_prediction.columns
    if ('soPhongTam' in cols):
        df_for_prediction['phongTam/dientich'] = df_for_prediction['soPhongTam'] / df_for_prediction['dienTich']
    if ('soPhongNgu' in cols):
        df_for_prediction['phongngu/dientich'] = df_for_prediction['soPhongNgu'] / df_for_prediction['dienTich']
    if ('soTang' in cols):
        df_for_prediction['Overall_S'] = df_for_prediction['soTang'] * df_for_prediction['dienTich']

    print(df_for_prediction.columns)

    prediction = model.predict(df_for_prediction[top10])

    df_after_prediction = df_for_prediction.copy()
    df_after_prediction['giaCa'] = prediction
    prediction_format = ['loai', 'ngayDangTin', 'giaCa']
    df_after_prediction = df_after_prediction[prediction_format]

    from datetime import date, timedelta

    trend_lst = []
    for typ in df_after_prediction['loai'].unique():
        fr1 = df_after_prediction['loai'] == typ
        fr2 = df_after_prediction['ngayDangTin'] == date.today()
        fr3 = df_after_prediction['ngayDangTin'] == date.today() + timedelta(weeks=12)

        trend_plateau = df_after_prediction[fr1][fr2].giaCa.values == df_after_prediction[fr1][fr3].giaCa.values
        trend_increase = df_after_prediction[fr1][fr2].giaCa.values < df_after_prediction[fr1][fr3].giaCa.values
        if trend_plateau[0]:
            trend_lst.append([typ, 'Không đổi'])
        elif trend_increase[0]:
            trend_lst.append([typ, 'Tăng'])
        else:
            trend_lst.append([typ, 'Giảm'])

    df_trend = pd.DataFrame(trend_lst).rename(columns={0: 'loai', 1: 'trend'})

    dict_final = []
    for i in range(len(df_trend)):
        dat = {}
        dat['id'] = i
        dat['loai']  = df_trend.loc[i, 'loai']
        dat['trend'] = df_trend.loc[i, 'trend']
        dat['reportData'] = []
        for idx in df_after_prediction[df_after_prediction['loai'] == df_trend.loc[i, 'loai']].index:
            dat['reportData'].append(
                {'date': df_after_prediction.loc[idx, 'ngayDangTin'], 'price': df_after_prediction.loc[idx, 'giaCa']})
        dict_final.append(dat)

    return dict_final
# def predict_trend_loai(quan, longitude, latitude, model, top10, data_cleaned):

#     data_district = data_cleaned["tinhThanhPho"].str.split(
#         ", ").map(lambda x: x[-2])

#     highest_match_district = 0
#     highest_accuracy = 0
#     districts = data_district.unique()
#     for i in range(len(districts)):
#         acc = fuzz.token_set_ratio(districts[i], quan)
#         if (acc > highest_accuracy):
#             highest_match_district = i
#             highest_accuracy = acc

#     quan = districts[highest_match_district]

#     house_types = list(data_cleaned[data_district == quan].loai.unique())
#     avg_col = []

#     for col in data_cleaned.columns:
#         if data_cleaned[col].dtypes != 'int64' and data_cleaned[col].dtypes != 'object' and data_cleaned[col].dtypes != 'datetime64[ns]':
#             avg_col.append(col)
#     avg_col.append('phapLy')
#     avg_col.remove('giaCa')

#     house_type_col = list(data_cleaned['loai'].unique())
#     # for house_type in house_type_col:
#     #   avg_col.remove(house_type)

#     bool_col = []
#     for col in data_cleaned.columns:
#         if data_cleaned[col].dtypes == 'bool':
#             bool_col.append(col)

#     data_new = data_cleaned.copy()
#     data_new[bool_col] = data_new[bool_col].astype('float64')
#     for col in data_new.columns:
#         if col in avg_col:
#             data_new[col].fillna(data_new[col].mean(), inplace=True)
#     avg_col_dat = list(np.average(data_new[avg_col], 0))

#     data_pred = []
#     today = date.today()
#     for house in house_types:
#         for i in range(5):
#             months_added = timedelta(weeks=12 * (i - 3))
#             row = []
#             row.append(house)
#             row.append(today + months_added)
#             for dat in avg_col_dat:
#                 row.append(dat)
#             data_pred.append(row)

#     lst_of_feature = ['loai', 'ngayDangTin'] + avg_col
#     df_for_prediction = pd.DataFrame(data_pred).rename(
#         columns=dict(enumerate(lst_of_feature)))

#     df_for_prediction['longitude'] = longitude
#     df_for_prediction['latitude'] = latitude

#     df_for_prediction['ngayDangTin_ts'] = df_for_prediction['ngayDangTin'].apply(
#         lambda x: mdates.date2num(x))

#     for house_type in house_type_col:
#         df_for_prediction[house_type] = 0

#     for house_type in house_types:
#         idx = df_for_prediction[df_for_prediction['loai'] == house_type].index
#         df_for_prediction.loc[idx, house_type] = 1


#     # create new features
#     cols = df_for_prediction.columns
#     if ('soPhongTam' in cols):
#         df_for_prediction['phongTam/dientich'] = df_for_prediction['soPhongTam'] / df_for_prediction['dienTich']
#     if ('soPhongNgu' in cols):
#         df_for_prediction['phongngu/dientich'] = df_for_prediction['soPhongNgu'] / df_for_prediction['dienTich']
#     if ('soTang' in cols):
#         df_for_prediction['Overall_S'] = df_for_prediction['soTang'] * df_for_prediction['dienTich']

#     print(df_for_prediction.columns)

#     try:
#         prediction = model.predict(df_for_prediction[top10])
#     except:
#         print('hahahahaha')


#     df_after_prediction = df_for_prediction.copy()
#     df_after_prediction['giaCa'] = prediction
#     prediction_format = ['loai', 'ngayDangTin', 'giaCa']
#     df_after_prediction = df_after_prediction[prediction_format]

#     trend_lst = []
#     for typ in df_after_prediction['loai'].unique():
#         fr1 = df_after_prediction['loai'] == typ
#         fr2 = df_after_prediction['ngayDangTin'] == date.today()
#         fr3 = df_after_prediction['ngayDangTin'] == date.today(
#         ) + timedelta(weeks=12)

#         trend_plateau = df_after_prediction[fr1][fr2].giaCa.values == df_after_prediction[fr1][fr3].giaCa.values
#         trend_increase = df_after_prediction[fr1][fr2].giaCa.values < df_after_prediction[fr1][fr3].giaCa.values
#         if trend_plateau[0]:
#             trend_lst.append([typ, 'Không đổi'])
#         elif trend_increase[0]:
#             trend_lst.append([typ, 'Tăng'])
#         else:
#             trend_lst.append([typ, 'Giảm'])

#     df_trend = pd.DataFrame(trend_lst).rename(columns={0: 'loai', 1: 'trend'})

#     dict_final = []
#     for i in range(len(df_trend)):
#         dat = {}
#         dat['id'] = i
#         dat['trend'] = df_trend.loc[i, 'trend']
#         dat['reportData'] = []
#         for idx in df_after_prediction[df_after_prediction['loai'] == df_trend.loc[i, 'loai']].index:
#             dat['reportData'].append(
#                 {'date': df_after_prediction.loc[idx, 'ngayDangTin'], 'price': df_after_prediction.loc[idx, 'giaCa']})
#         dict_final.append(dat)

#     return dict_final


def predict_future(model, top10, cleaned_data, long, lat, loai, dienTich, soTang, soPhongNgu, soPhongTam, phapLy, dacDiemXahoi, tienIchKemTheo, noiThat):
    list_dd = ['Gần trường', 'Gần bệnh viện', 'Gần công viên',
               'Gần nhà trẻ', 'Tiện kinh doanh', 'Khu dân trí cao',
               'Gần chợ']
    list_ti = ['Chỗ để xe máy', 'Chỗ để ôtô', 'Trung tâm thể dục', 'Hệ thống an ninh',
               'Nhân viên bảo vệ', 'Hồ bơi', 'Truyền hình cáp', 'Internet']
    list_noiThat = ['Bàn ăn', 'Bàn trà',  'Sofa phòng khách', 'Kệ ti vi', 'Giường ngủ', 'Tủ quần áo',
                    'Sàn gỗ/đá', 'Trần thả', 'Tủ bếp',
                    'Bình nóng lạnh', 'Điều hòa', 'Bồn rửa mặt', 'Bồn tắm']
    house_type_col = ['Nhà mặt phố', 'Nhà riêng', 'Căn hộ Cao cấp', 'Nhà phố Shophouse', 'Biệt thự liền kề', 'Nhà trọ, phòng trọ', 'Bất động sản khác', 'Căn hộ trung cấp', 'Căn hộ chung cư',
                      'Nhà biệt thự', 'Căn hộ rẻ', 'Nhà rẻ', 'Cửa hàng kiot', 'Căn hộ Tập thể', 'Căn hộ mini', 'Khách sạn', 'Biệt thự nghỉ dưỡng', 'Căn hộ Officetel', 'Mặt bằng bán lẻ', 'Căn hộ Penthouse', 'Văn phòng']
    df = pd.DataFrame()
    data_pred = []
    today = date.today()
    data_pred.append(today)
    for i in range(2):
        months_added = timedelta(weeks=4 * i)
        data_pred.append(today + months_added)
    df['ngayDangTin'] = data_pred
    df['ngayDangTin_ts'] = df['ngayDangTin'].apply(
        lambda x: mdates.date2num(x))
    df['longitude'] = float(long)
    df['latitude'] = float(lat)
    df['loai'] = loai
    for i in house_type_col:
        if i == loai:
            df[i] = 1
        else:
            df[i] = 0
    if dienTich is not None:
        df['dienTich'] = int(dienTich)
    else:
        df['dienTich'] = cleaned_data[cleaned_data['loai']
                                      == loai]['dienTich'].mean()
    if soTang is not None:
        df['soTang'] = int(soTang)
    else:
        df['soTang'] = round(
            cleaned_data[cleaned_data['loai'] == loai]['soTang'].mean(), 0)
    if soPhongNgu is not None:
        df['soPhongNgu'] = int(soPhongNgu)
    else:
        df['soPhongNgu'] = round(
            cleaned_data[cleaned_data['loai'] == loai]['soPhongNgu'].mean(), 0)
    if soPhongTam is not None:
        df['soPhongTam'] = int(soPhongTam)
    else:
        df['soPhongTam'] = round(
            cleaned_data[cleaned_data['loai'] == loai]['soPhongTam'].mean(), 0)
    if phapLy is not None:
        df['phapLy'] = int(phapLy)
    else:
        df['phapLy'] = 0
    for i in list_dd:
        if i in dacDiemXahoi:
            df[i] = 1
        else:
            df[i] = 0
    for i in list_ti:
        if i in tienIchKemTheo:
            df[i] = 1
        else:
            df[i] = 0
    for i in list_noiThat:
        if i in noiThat:
            df[i] = 1
        else:
            df[i] = 0
    df['phongTam/dientich'] = df['soPhongTam']/df['dienTich']
    df['phongngu/dientich'] = df['soPhongNgu']/df['dienTich']
    df['Overall_S'] = df['soTang']*df['dienTich']
    y_pred_1 = model.predict(df[top10])

    prediction = model.predict(df[top10])

    df_after_prediction = df.copy()
    df_after_prediction['giaCa'] = prediction
    prediction_format = ['loai', 'ngayDangTin', 'giaCa']
    df_after_prediction = df_after_prediction[prediction_format]

    trend_lst = []
    for typ in df_after_prediction['loai'].unique():
        fr1 = df_after_prediction['loai'] == typ
        fr2 = df_after_prediction['ngayDangTin'] == date.today()
        fr3 = df_after_prediction['ngayDangTin'] == date.today(
        ) + timedelta(weeks=4)

    trend_plateau = df_after_prediction[fr1][fr2].giaCa.values == df_after_prediction[fr1][fr3].giaCa.values
    trend_increase = df_after_prediction[fr1][fr2].giaCa.values < df_after_prediction[fr1][fr3].giaCa.values
    if trend_plateau[0]:
        trend_lst.append([typ, 'Không đổi'])
    elif trend_increase[0]:
        trend_lst.append([typ, 'Tăng'])
    else:
        trend_lst.append([typ, 'Giảm'])

    df_trend = pd.DataFrame(trend_lst).rename(columns={0: 'loai', 1: 'trend'})

    dict_final = []
    for i in range(len(df_trend)):
        dat = {}
        dat['id'] = i
        dat['trend'] = df_trend.loc[i, 'trend']
        dat['reportData'] = []
        for idx in df_after_prediction[df_after_prediction['loai'] == df_trend.loc[i, 'loai']].index:
            dat['reportData'].append(
                {'date': df_after_prediction.loc[idx, 'ngayDangTin'], 'price': df_after_prediction.loc[idx, 'giaCa']})
        dict_final.append(dat)

    return dict_final


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/getAll')
def getAll():
    csvFilePath = r"clean.csv"
    jsonFilePath = r'all.json'
    make_json(csvFilePath, jsonFilePath)
    with open('all.json') as f:
        data = json.load(f)
    return data


@app.route('/get30closest/<latitude>/<longitude>')
def get30closest(latitude, longitude):
    neighbor = neighbor_detection(latitude, longitude, data_cleaned)
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


@app.route('/predictByType/<quan>/<longitude>/<latitude>')
def predictByType(quan, longitude, latitude):
    result = predict_trend_loai(
        quan, longitude, latitude, model, top10, data_cleaned)
    result_df = pd.DataFrame(result)
    result_df.to_csv("predict.csv")
    csvFilePath = r"predict.csv"
    jsonFilePath = r'predict.json'
    make_json(csvFilePath, jsonFilePath)
    with open('predict.json') as f:
        data = json.load(f)
    return data


@app.route('/predictHouse/<longitude>/<lat>/<loai>/<dienTich>/<soTang>/<soPhongNgu>/<soPhongTam>/<phapLy>/<dacDiemXahoi>/<tienIchKemTheo>/<noiThat>')
def predictHouse(longitude, lat, loai, dienTich, soTang, soPhongNgu, soPhongTam, phapLy, dacDiemXahoi, tienIchKemTheo, noiThat):
    print(type(loai), loai)
    result = predict_future(model, top10, data_cleaned, longitude, lat, loai, dienTich,
                            soTang, soPhongNgu, soPhongTam, phapLy, dacDiemXahoi, tienIchKemTheo, noiThat)
    result_df = pd.DataFrame(result)
    result_df.to_csv("predictHouse.csv")
    csvFilePath = r"predictHouse.csv"
    jsonFilePath = r'predictHouse.json'
    make_json(csvFilePath, jsonFilePath)
    with open('predictHouse.json') as f:
        data = json.load(f)
    return data



if __name__ == "__main__":
    app.run(debug=True)
