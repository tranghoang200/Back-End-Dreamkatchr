import numpy as np
from flask import Flask, request, jsonify, render_template
import json
import pandas as pd
import csv 

url = 'https://raw.githubusercontent.com/QuocAnh261100/GQA/master/data_long_lat.csv'
raw_data = pd.read_csv(url)


# with open('test1.json') as f:
#  data = json.load(f)


app = Flask(__name__)

def geocode_transformation(location): # Location taken in the form of columns tinhThanhPho
  from geopy.geocoders import Nominatim
  geolocator = Nominatim(user_agent = "MyEncoder")
  split_ = location.split(', ')
  geocode_trans = None
  for i in range(len(split_) - 1):
    loc = ', '.join(split_[i:])
    geocode_trans = geolocator.geocode(loc)
    if geocode_trans is not None:
      return geocode_trans
  return geocode_trans

def distance(lat_1, long_1, lat_2, long_2):
  import geopy.distance
  return geopy.distance.great_circle((lat_1, long_1), (lat_2, long_2))

def neighbor_dectection_with_longlat(geocode_trans, data):
  distances = []
  long_ = geocode_trans.longitude
  lat_ = geocode_trans.latitude
  for i in range(len(data)):
    distances.append(distance(lat_, long_, data.loc[i].latitude, data.loc[i].longitude))
  distances = np.array(distances)
  return data.loc[np.argsort(distances)[:20]]


def neighbor_detection(address, data):
  geocode_trans = geocode_transformation(address)
  return neighbor_dectection_with_longlat(geocode_trans, data)

# Function to convert a CSV to JSON 
# Takes the file paths as arguments 
def make_json(csvFilePath, jsonFilePath): 
	
	# create a dictionary 
	data = {} 
	
	# Open a csv reader called DictReader 
	with open(csvFilePath, encoding='utf-8') as csvf: 
		csvReader = csv.DictReader(csvf) 
		
		# Convert each row into a dictionary 
		# and add it to data 
		for rows in csvReader: 
			
			# Assuming a column named 'No' to 
			# be the primary key 
			key = rows[''] 
			data[key] = rows 

	# Open a json writer, and use the json.dumps() 
	# function to dump data 
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
    
@app.route('/get20closest/<address>')
def get20closest(address):
    neighbor = neighbor_detection(address, raw_data)
    neighbor.to_csv("neighbor.csv")
    csvFilePath = r'neighbor.csv'
    jsonFilePath = r'result.json'
    make_json(csvFilePath, jsonFilePath)
    with open('result.json') as f:
        data = json.load(f)
    return data

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    pass


if __name__ == "__main__":
    app.run(debug=True)