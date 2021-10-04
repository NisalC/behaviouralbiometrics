from flask import Flask,request
import pandas as pd
from _collections import OrderedDict
import joblib

app=Flask(__name__)


@app.route('/api/bimetrics')
def get():
    duration=float(request.args['duration'])
    press_press_latency=float(request.args['ppl'])
    release_press_latency=float(request.args['rpl'])
    press_pressure=float(request.args['pp'])
    press_touch_major=float(request.args['ptmajor'])
    press_touch_minor=float(request.args['ptminor'])
    press_x=float(request.args['pr_x'])
    press_y=float(request.args['pr_y'])
    press_x_acceleration=float(request.args['pr_x_acc'])
    press_y_acceleration=float(request.args['pr_y_acc'])
    press_z_acceleration=float(request.args['pr_z_acc'])
    press_x_rotation=float(request.args['pr_x_rot'])
    press_y_rotation=float(request.args['pr_y_rot'])
    press_z_rotation=float(request.args['pr_z_rot'])

    folder = 'behaviorModel/'
    filePath = folder + 'behavior_model.joblib'
    file = open(filePath, "rb")
    trained_model = joblib.load(file)

    data=OrderedDict([('duration',duration),('press_press_latency',press_press_latency),('release_press_latency',release_press_latency),('press_pressure',press_pressure),('press_touch_major',press_touch_major),('press_touch_minor',press_touch_minor),('press_x',press_x),('press_y',press_y),('press_x_acceleration',press_x_acceleration),('press_y_acceleration',press_y_acceleration),('press_z_acceleration',press_z_acceleration),('press_x_rotation',press_x_rotation),('press_y_rotation',press_y_rotation),('press_z_rotation',press_z_rotation)])
    data=pd.Series(data).values.reshape(1,-1)
    prediction = trained_model.predict(data)
    return str(prediction)

if __name__ == '__main__':
    app.run()
