import pickle
from tensorflow import keras
from flask import Flask,render_template,request
import pandas as pd
import numpy as np

model = keras.models.load_model('model/model.h5')
ohe = pickle.load(open('model/ohe.pkl','rb'))
venue_encoder = pickle.load(open('model/venue_encoder.pkl','rb'))
team_encoder = pickle.load(open('model/team_encoder.pkl','rb'))
batting_avg = pickle.load(open('model/batting_avg.pkl','rb'))
bowling_avg = pickle.load(open('model/bowling_avg.pkl','rb'))
venue_avg = pickle.load(open('model/venue_avg.pkl','rb'))
scaler = pickle.load(open('model/scaler.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    venue_list = venue_avg['venue'].values.tolist()
    team_list = batting_avg['bat_team'].values.tolist()
    return render_template("index.html", venue_list = venue_list,team_list=team_list)

@app.route('/predict',methods=['post'])
def predict():

    venue = request.form.get('venue')
    batting_team = request.form.get('batting_team')
    bowling_team = request.form.get('bowling_team')
    runs_now = request.form.get('runs_now')
    wickets_now = request.form.get('wickets_now')
    overs_now = request.form.get('overs_now')
    runs_last5 = request.form.get('runs_last5')
    wickets_last5 = request.form.get('wickets_last5')

    venue_avg_score = int(venue_avg[venue_avg['venue']==venue]['avg_score_stadium'].values[0])

    batting_team_avg_score = int(batting_avg[batting_avg['bat_team']==batting_team]['total'].values[0])

    bowling_team_avg_score = int(bowling_avg[bowling_avg['bowl_team'] == bowling_team]['total'].values[0])

    df_input = [[venue,batting_team,bowling_team,runs_now,wickets_now,overs_now,runs_last5,wickets_last5,venue_avg_score,batting_team_avg_score,bowling_team_avg_score]]

    innings = pd.DataFrame(df_input, columns=['venue','batting_team','bowling_team','runs','wickets','overs','run_last5','wickets_last5','venue_avg_score','batting_team_avg_score','bowling_team_avg_score'])

    innings['venue'] = venue_encoder.transform(innings['venue'])
    innings['batting_team'] = team_encoder.transform(innings['batting_team'])


    innings['bowling_team'] = team_encoder.transform(innings['bowling_team'])

    X = innings.drop(['venue', 'batting_team', 'bowling_team'], axis=1)

    X_trans = ohe.transform(innings[['venue', 'batting_team', 'bowling_team']]).toarray()

    X = np.hstack((X, X_trans))

    X = scaler.transform(X)

    y_pred = model.predict(X)

    score = int(y_pred[0][0])

    venue_list = venue_avg['venue'].values.tolist()
    team_list = batting_avg['bat_team'].values.tolist()

    return render_template("index.html",score=score,venue_list = venue_list,team_list=team_list)


if __name__ == "__main__":
    app.run(debug=True)

