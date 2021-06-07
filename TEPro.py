from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from bs4 import BeautifulSoup
import requests

def get_values(name):
	my_dict = {
		"Delhi": "https://aqicn.org/city/delhi",
		"Lucknow": "https://aqicn.org/city/india/lucknow/talkatora/",
		"Ahmedabad": "https://aqicn.org/city/india/ahmedabad/maninagar/",
		"Kanpur": "https://aqicn.org/city/kanpur",
		"Faridabad": "https://aqicn.org/city/india/faridabad/sector-11/"
	}
	source = requests.get(my_dict[name]).text
	soup = BeautifulSoup(source, 'lxml')
	m1 = soup.find('td', id="cur_pm25").text
	m2 = soup.find('td', id="cur_no2").text
	m3 = soup.find('td', id="cur_pm10").text
	l = np.array([int(m2),int(m1), int(m3)]).reshape(1,-1)
	return l


def prediction(name, crop, area):
	r2 = 0
	data = pd.read_csv("/home/ian/Desktop/UbuntuShare/{}.csv".format(name))

	x = np.array(data[["no2", "rspm", "spm"]])
	y = np.array(data["{}_yield".format(crop)])
	for i in range(500):
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
		rf = GradientBoostingRegressor()
		rf.fit(x_train, y_train)
		r2 = r2_score(y_test, rf.predict(x_test))
		if((1-r2)*100 < 0.5):
			break

	y_pred = area * rf.predict(get_values(name))
	score1 = mean_squared_error(y_test, rf.predict(x_test))

	return y_pred[0], score1*area



app = Flask(__name__, static_folder='dist')
@app.route('/', methods=["POST", "GET"])
def home():
	if request.method == "POST":
		user1 = float(request.form["var1"])
		user2 = request.form["var2"]
		user3 = request.form["var3"]
		x,y = prediction(user2,user3,user1)

		return render_template('prediction.html', prod = "{:.2f}".format(x), prod2=user2, prod3 = user3, prod4="{:.2f}".format(y))
	else:
		return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, port=5000)
