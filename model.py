import pandas as pd
import requests
from fuzzywuzzy import fuzz
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.dates as mdates
from sklearn.cluster import DBSCAN

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
import matplotlib.pyplot as plt
import xgboost
from datetime import date, timedelta


url = 'https://raw.githubusercontent.com/QuocAnh261100/GQA/master/csvjson_1.csv'
raw_data = pd.read_csv(url)

# 2. CLEANING DATA


def clean(raw_data):
    # raw_data: data without any modification
    # data: data with modification

    # eliminate instances that are not "Nha"
    ax = raw_data.loai.str.contains("Đất")
    data = raw_data[~ax]
    data.loai.value_counts()

    # Drop "Cho Thue" instances (which contains 'thang')
    dro = data[data.giaCa.str.contains("tháng")].index
    data.drop(index=dro, inplace=True)

    # loại bỏ các instance giá cả thỏa thuận
    dro1 = data[data['giaCa'].str.contains("Thỏa thuận")].index
    data.drop(index=dro1, inplace=True)

    # re_index data 1 để giải quyết một số vấn đề về iteration về sau
    data.reset_index(drop=True, inplace=True)

    # loại bỏ m^2 trong cột value diện tích và ép kiểu thành float
    data['dienTich'] = data['dienTich'].str.replace(' m\u00B2', '', regex=True)
    data['dienTich'] = data['dienTich'].str.replace(',', '.', regex=True)
    data['dienTich'] = data['dienTich'].astype(float)

    # xử lí các dữ liệu mà giá nhà có ghi Triệu/m^2
    id_trieu = data[data.giaCa.str.contains("Triệu")].index
    idd = data[data.giaCa.str.contains("Triệu/")].index
    data['giaCa'] = data.giaCa.str.replace('Triệu[/]m\u00B2', '', regex=True)

    # xử lí dữ liệu giá nhà có chữ Triệu
    data['giaCa'] = data['giaCa'].str.replace(' Triệu', '', regex=True)
    data['giaCa'] = data.giaCa.str.replace(',', '.', regex=True)

    # xử lí dữ liệu giá nhà có chữ Tỷ
    data['giaCa'] = data['giaCa'].str.replace(' Tỷ', '', regex=True)
    data['giaCa'] = data.giaCa.str.replace(',', '.', regex=True)

    # Xử lý số tầng
    # Trong mô tả sẽ ghi là 'nhà ... tầng' nên từ đó ta tìm ra số tầng
    id1 = data.loc[data.moTa.str.contains("[0-9]+ [tT][ầẦ][Nn][Gg]")].index

    # Tìm các instance mà số Tầng k là nan
    data.soTang.fillna(0, inplace=True)
    id_not_nan = data[data.soTang != 0].index

    # Loại bỏ các instance trong id1 mà đã có số tầng
    id1 = id1.difference(id_not_nan)
    tang_new = data.loc[id1].moTa.str.findall(
        '[0-9]+ [tT][ầẦ][Nn][Gg]').map(lambda x: x[0])
    tang_new_1 = tang_new.str.split(' ').map(lambda x: x[0]).astype(float)
    data.at[id1, 'soTang'] = tang_new_1
    qa = data[data.loai.str.contains('Căn hộ')].soTang
    id_tang_rep = qa[qa >= 3].index
    id_tang_rep = id_tang_rep.append(qa[qa == 0].index)
    cre = np.ones((len(id_tang_rep), 1))

    # Thay các instance có số nhà bằng 0 thành giá trị trung bình
    data.at[id_tang_rep, 'soTang'] = cre
    id_not_nan = data[data.soTang != 0].index
    id_nan = data.index.difference(id_not_nan)
    a = round(data.loc[id_not_nan].soTang.mean(), 0)
    for i in id_nan:
        data.at[i, 'soTang'] = a

    # Xử lý dữ liệu nhà triệu/m^2
    df = data.loc[idd]
    df['giaCa'] = df['giaCa'].astype(float)

    # các nhà Căn hộ có số tầng > 3 là các giá trị sai và sửa số tầng thành 1
    sotang_wrong = df[df['loai'].str.contains(
        "Căn hộ")][df['soTang'] >= 3].index
    df.at[sotang_wrong, 'soTang'] = 1

    # also fix in data
    data.at[sotang_wrong, 'soTang'] = 1

    # vì diện tích chỉ là diện tích 1 tầng nên giá nhà phải nhân với số tầng
    giaca_new = df['giaCa']*df['dienTich']*df['soTang']/1000

    data.at[idd, 'giaCa'] = giaca_new

    # biến đổi đơn vị triệu xuống tỷ
    data['giaCa'] = data['giaCa'].astype(float)
    for i in id_trieu:
        if i not in idd:
            data.loc[i, 'giaCa'] = data.loc[i, 'giaCa']/1000

    # convert date time to Date format.
    data['ngayDangTin'] = pd.to_datetime(
        data['ngayDangTin'], format='%d/%m/%Y')
    data['ngayHetHan'] = pd.to_datetime(data['ngayHetHan'], format='%d/%m/%Y')

    # xử lí feature pháp lý: nếu là nan -> 0; nếu không phải nan -> 1
    id_phaply = data['phapLy'].str.match('[a-zA-Z]+').dropna().index
    data['phapLy'] = 0
    data.at[id_phaply, 'phapLy'] = 1

    # get long_lat
    geocodes = []

    for i in range(len(data)):
        raw = data['diaChi'][i] + ', ' + data['tinhThanhPho'][i]
        done = raw.replace(' ', '+')
        response = requests.get('https://maps.googleapis.com/maps/api/geocode/json?address=' +
                                done + '&key=AIzaSyAhuvkbu8iQU3vptKQSbaHQNlTJv0ndTVw')
        dataJson = response.json()
        if (dataJson['status'] != 'OK'):
            location = {'lat': None, 'lng': None}
        else:
            location = dataJson['results'][0]['geometry']['location']
        geocodes.append(location)

    data['latitude'] = [g['lat'] for g in geocodes]
    data['longitude'] = [g['lng'] for g in geocodes]

    # Xử lý các tiện ích kèm theo.
    # Chon tienIchKemTheo dai nhat roi trich xuat cac thuoc tinh
    ax = data.tienIchKemTheo.map(lambda x: len(str(x)))
    id_tien = ax[ax == ax.max()].index
    df2 = data.loc[id_tien].tienIchKemTheo
    str_max = df2[id_tien[0]]
    ti = str_max.split(',')
    ti = ['Chỗ để xe máy', 'Chỗ để ôtô', 'Trung tâm thể dục', 'Hệ thống an ninh',
          'Nhân viên bảo vệ', 'Hồ bơi', 'Truyền hình cáp', 'Internet']

    for i in ti:
        data[i] = data.tienIchKemTheo.str.contains(i).fillna(False).values

    for i in range(len(ti)):
        id_1 = data[data[ti[i]] == False].index
        s = data.loc[id_1].moTa
        g = s.map(lambda x: fuzz.token_set_ratio(x, ti[i]))
        id_2 = (g > 60).index
        data.at[id_2, ti[i]] = g > 60

    # Xử lý các đặc điểm xã hội
    dx = ['Gần trường', 'Gần bệnh viện', 'Gần công viên',
          'Gần nhà trẻ', 'Tiện kinh doanh', 'Khu dân trí cao', 'Gần chợ']

    for i in dx:
        data[i] = data.dacDiemXaHoi.str.contains(i).fillna(False)

    # sử dụng fuzzy search để tìm kiếm các dac diem xa hoi trong phần mô tả
    for i in range(len(dx)):
        id_1 = data[data[dx[i]] == False].index
        s = data.loc[id_1].moTa
        g = s.map(lambda x: fuzz.token_set_ratio(x, ti[i]))
        id_2 = (g > 60).index
        data.at[id_2, dx[i]] = g > 60
    # xử lý nội thất
    dx1 = ['Bàn ăn', 'Bàn trà', 'Sofa phòng khách', 'Kệ ti vi', 'Giường ngủ', 'Tủ quần áo',
           'Sàn gỗ/đá', 'Trần thả', 'Tủ bếp', 'Bình nóng lạnh', 'Điều hòa', 'Bồn rửa mặt', 'Bồn tắm']
    for i in dx1:
        data[i] = data.noiThat.str.contains(i).fillna(False)

    # phaply processing
    id_sd = data[data.moTa.str.contains('[sS][ổỔ] [đĐ][ỏỎ]')].index
    len(id_sd)
    id_phaply = data[data['phapLy'] == 1].index
    data['phapLy'] = 0
    data.at[id_phaply, 'phapLy'] = 1

    # loai nha
    loai_encoder = OneHotEncoder()
    loai_1hot = loai_encoder.fit_transform(data[['loai']])

    df = pd.DataFrame(loai_1hot.toarray(), columns=loai_encoder.categories_[0])
    data = pd.concat([data, df], axis=1)
    data.to_csv('clean.cvs')
    # ma tin

    return data


# 3. Preprocessing data before training
def preprocessing(unpreprocessed_data):
    data = unpreprocessed_data.copy()
    # missing data handle
    is_null = ((data.isnull().sum())/len(data)).sort_values(ascending=False)
    drop_col = is_null[is_null > 0.5].index.to_list()
    # drop_col.append('loai')
    data = data.drop(columns=drop_col)

    # convert time to TimeSeries
    data['ngayDangTin_ts'] = data['ngayDangTin'].apply(
        lambda x: mdates.date2num(x))
    data['ngayHetHan_ts'] = data['ngayHetHan'].apply(
        lambda x: mdates.date2num(x))

    # use DBSCAN to find outliers
    outlier_detection = DBSCAN(
        eps=7, metric='euclidean', min_samples=5, n_jobs=-1)
    used_col = []
    for col in data.columns:
        if data[col].dtypes != 'int64' and data[col].dtypes != 'object' and data[col].dtypes != 'datetime64[ns]':
            used_col.append(col)

    # fillna missing value
    for i in used_col:
        data[i].fillna(value=round(data[i].mean(), 0), inplace=True)

    # data.dropna(subset = used_col, inplace = True)

    clusters = pd.Series(outlier_detection.fit_predict(data[used_col]))

    frame = pd.Series(clusters)
    idx = frame[frame == -1].index

    # fix wrong giaCa data
    gia = data.loc[idx].moTa.str.findall(
        '[0-9]+[,.]{0,1}[0-9]*[ ]*[Tt][ỷỶ]|[0-9]+[,.]{0,1}[0-9]*[ ]*[Tt][ỉỈ]')
    iiii = data.loc[idx].moTa.str.findall(
        '[0-9]+[,.]{0,1}[0-9]*[ ]*[Tt][ỷỶ]|[0-9]+[,.]{0,1}[0-9]*[ ]*[Tt][ỉỈ]').map(lambda x: len(x) == 1 or len(x) == 2 or len(x) == 3)
    id_rep_gia = gia[iiii].index
    series_gia_new = gia[iiii].map(lambda x: x[0]).str.findall(
        '[0-9]+[,.]*[0-9]*').map(lambda x: x[0]).str.replace(',', '.')

    data.at[id_rep_gia, 'giaCa'] = series_gia_new
    data['giaCa'] = data['giaCa'].astype(float)

    # eliminate columns with data type 'object'
    col_not_object = data.dtypes[data.dtypes != 'object'].index
    data = data[col_not_object]

    data['dienTichXsoTang'] = data['dienTich'] * data['soTang']

    # create new features
    cols = data.columns
    if ('soPhongTam' in cols):
        data['phongTam/dientich'] = data['soPhongTam']/data['dienTich']
    if ('soPhongNgu' in cols):
        data['phongngu/dientich'] = data['soPhongNgu']/data['dienTich']
    if ('soTang' in cols):
        data['Overall_S'] = data['soTang']*data['dienTich']

    # # drop instance irrelevant
    # list_drop = ['Bất động sản khác','Nhà phố Shophouse','Căn hộ mini','Văn phòng',
    #            'Biệt thự nghỉ dưỡng','Cửa hàng kiot','Căn hộ Officetel','Khách sạn']
    # for i in list_drop:
    #   data.drop(index = data[data[i] == 1].index, inplace = True)
    data.to_csv('preprocess.csv')
    return data


# 4. Train data (dataFrame)
def train(data):

    # Random Forest Feature Importance
    def Feature_Importance(df, fea_cols, target_col, a):
        X = df[fea_cols]
        y = df[target_col]
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor()
        model.fit(X, y)
        importance = model.feature_importances_
        argsort_idx = np.argsort(importance)[::-1]
        for i in argsort_idx:
            print('Feature: '+str(fea_cols[i]) +
                  ', Score: %.5f' % (importance[i]))
        # Lấy 1 số feature ảnh hưởng nhất đến giá cả
        _ = []
        for i in argsort_idx[:a]:
            _.append(fea_cols[i])
        return _
    data_ = data.drop(columns=['ngayDangTin', 'ngayHetHan', 'maTin'])
    top10 = Feature_Importance(data_, data_.columns[1:], data_.columns[0], 10)
    X = data[top10]
    y = data['giaCa']
    # split train and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True)
    X_train = X_train.sort_index()
    y_train = y_train.sort_index()
    X_test = X_test.sort_index()
    y_test = y_test.sort_index()
    # optimze parameters for xgboost model
    param_grid = [{'max_depth': [3, 5, 7, 9], 'min_child_weight':[1, 2, 3, 4], 'objective': ['reg:linear'], 'colsample_bytree': [0.6, 0.8, 1], 'learning_rate': [0.001, 0.01],
                   'reg_lambda': [0.1, 0.5], 'n_estimators': [100, 1000]}]
    model = xgboost.XGBRegressor()
    grid_search = GridSearchCV(model, param_grid, cv=7,
                               scoring='neg_mean_squared_error',
                               return_train_score=True, refit=True)
    grid_search.fit(X_train, y_train)
    xgb_reg = grid_search.best_estimator_
    xgb_reg.fit(X_train, y_train, verbose=1)
    pickle.dump(xgb_reg, open("pima.pickle.dat", "wb"))
    # calculate rmse and mape of model
    y_pred_xgb = xgb_reg.predict(X_test)
    xgb_mse = (y_test - y_pred_xgb)**2/len(y_test)
    xgb_rmse = np.sqrt(xgb_mse)

    def MAPE(y_real, y_pred):
        return pd.Series(abs(np.array(y_real)-np.array(y_pred))/np.array(y_real)*100)
    xgb_mape = MAPE(y_test, y_pred_xgb)
    # save model and error
    pickle.dump(xgb_reg, open("pima.pickle.dat", "wb"))
    df = pd.DataFrame(data=[xgb_rmse, xgb_mape], columns=[
                      'error'], index=['xgb_rmse', 'xgb_mape'])
    df.to_csv('error.csv', index=False)
    return xgb_reg, xgb_rmse, xgb_mape

# 5. Prediction


def predict_trend_loai(quan, longitude, latitude, model, top10, data_cleaned):

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
        if data_cleaned[col].dtypes != 'int64' and data_cleaned[col].dtypes != 'object' and data_cleaned[col].dtypes != 'datetime64[ns]':
            avg_col.append(col)
    avg_col.append('phapLy')
    avg_col.remove('giaCa')

    house_type_col = ['Căn hộ Tập thể', 'Nhà biệt thự',
                      'Nhà mặt phố', 'Nhà riêng', 'Nhà rẻ']
    # for house_type in house_type_col:
    #   avg_col.remove(house_type)

    bool_col = []
    for col in data_cleaned.columns:
        if data_cleaned[col].dtypes == 'bool':
            bool_col.append(col)

    data_new = data_cleaned.copy()
    data_new[bool_col] = data_new[bool_col].astype('float64')
    for col in data_new.columns:
        if col in avg_col:
            data_new[col].fillna(data_new[col].mean(), inplace=True)
    avg_col_dat = list(np.average(data_new[avg_col], 0))

    data_pred = []
    for house in house_types:
        today = date.today()
        for i in range(6):
            months_added = timedelta(weeks=24 * i)
            row = []
            row.append(house)
            row.append(today + months_added)
            for dat in avg_col_dat:
                row.append(dat)
            data_pred.append(row)

    lst_of_feature = ['loai', 'ngayDangTin'] + avg_col
    df_for_prediction = pd.DataFrame(data_pred).rename(
        columns=dict(enumerate(lst_of_feature)))

    df_for_prediction['longitude'] = longitude
    df_for_prediction['latitude'] = latitude

    df_for_prediction['ngayDangTin_ts'] = df_for_prediction['ngayDangTin'].apply(
        lambda x: mdates.date2num(x))

    for house_type in house_types:
        idx = df_for_prediction[df_for_prediction['loai'] == house_type].index
        df_for_prediction.loc[idx, house_type] = 1
        others = house_type_col.copy()
        others.remove(house_type)
        df_for_prediction.loc[idx, others] = 0

    df_for_prediction['dienTichXsoTang'] = df_for_prediction['dienTich'] * \
        df_for_prediction['soTang']

    # create new features
    cols = df_for_prediction.columns
    if ('soPhongTam' in cols):
        df_for_prediction['phongTam/dientich'] = df_for_prediction['soPhongTam'] / \
            df_for_prediction['dienTich']
    if ('soPhongNgu' in cols):
        df_for_prediction['phongngu/dientich'] = df_for_prediction['soPhongNgu'] / \
            df_for_prediction['dienTich']
    if ('soTang' in cols):
        df_for_prediction['Overall_S'] = df_for_prediction['soTang'] * \
            df_for_prediction['dienTich']

    prediction = model.predict(df_for_prediction[top10])

    df_after_prediction = df_for_prediction.copy()
    df_after_prediction['giaCa'] = prediction
    prediction_format = ['loai', 'ngayDangTin', 'giaCa']
    df_after_prediction = df_after_prediction[prediction_format]

    return df_after_prediction


clean_data = clean(raw_data)
preprocess_data = preprocessing(clean_data)
train(preprocess_data)
