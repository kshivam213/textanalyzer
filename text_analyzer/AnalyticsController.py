from flask import Flask
from flask import request
from flask import Response
import TrainService
import TextAnalyticsService
import json

app = Flask(__name__)
@app.route('/api/v1/train/sentiment', methods = ['GET'])
def textAnalyticsTrain():
	trainService = TrainService.Train()
	return trainService.train()

@app.route('/api/v1/classify', methods = ['POST'])
def analyseText():
	text = request.get_json()
	textAnalyticsService= TextAnalyticsService.Classifier()
	return textAnalyticsService.analyze(text)

if __name__ == '__main__':
	app.run(host="0.0.0.0", debug=None)