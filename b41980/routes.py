from flask import request, render_template, url_for, flash, redirect
import joblib
from joblib import load
import numpy as np
import pandas as pd
import altair as alt
from b41980.forms import HouseForm
from b41980.models import House
from b41980 import app, db
from b41980.ml_helpers import alt_avenir

# Altair settings
alt.themes.register('alt_avenir', alt_avenir)
alt.themes.enable('alt_avenir')
alt.data_transformers.disable_max_rows()
props = {'width': 400, 'height': 300}

# Loading in our trained model and data
rf = load("b41980/model/rf_before1980.joblib")
lle_pipe = load("b41980/model/lle.joblib")
lle_data = pd.read_json("b41980/data/lle_chart_data.json")

# Creating api endpoints


@app.route('/isAlive')
def isAlive():
    return render_template("isAlive.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/contact')
def contact():
    return render_template("contact.html")


# @app.route('/predictapi')
# def predictapi():
#     # Reading all necessary request params
#     livearea = request.args.get("livearea")
#     stories = request.args.get("stories")
#     bdrms = request.args.get("bdrms")
#     baths = request.args.get("baths")

#     # Creating feature array
#     new_instance = np.array([[livearea, stories, bdrms, baths]])

#     # Predicting new_instance target
#     prediction = rf.predict(new_instance)

#     # Returning results back
#     return f"Predicted result for {new_instance} is: {prediction}"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Creating the form that allows users to input their house info
    form = HouseForm()

    # Validating form submission
    if form.validate_on_submit():
        # Getting information from the form and instantiating a row in the db
        house = House(livearea=form.livearea.data, bdrms=form.bdrms.data,
                      baths=form.baths.data, one_story=form.one_story.data,
                      att_garage=form.att_garage.data, basement=form.basement.data)
        # Adding row to db - think of comitting in version control
        db.session.add(house)
        # Pushing the new row to the actual db
        db.session.commit()
        # Sending a flash message saying the form was submitted
        flash('House information processed!', category='success')
        return redirect(url_for('predictresults'))

    # Returning results back
    return render_template("predict.html", title="Predict", form=form)


@app.route('/predictresults')
def predictresults():
    h1 = House.query.order_by(House.id.desc()).first()

    # Creating feature array
    new_instance = np.array([[h1.livearea, h1.bdrms, h1.baths, h1.one_story,
                              h1.att_garage, h1.basement]])

    # Predicting new_instance target
    prediction = rf.predict(new_instance)

    # Principal Components
    lle_new_instance = lle_pipe.transform(new_instance)
    lle_new_data = pd.DataFrame(lle_new_instance, columns=['pca1', 'pca2'])

    # Returning text depending on the prediction
    pred = "not" if prediction[0] == 0 else ""

    # Creating a sweet altair chart
    lle_chart = alt.Chart(lle_data).mark_circle(opacity=.4).encode(
        alt.X("pca1", title="principal component 1"),
        alt.Y("pca2", title="principal component 2"),
        alt.Color("target:N", title='Built before 1980', scale=alt.Scale(range=["gray", "orange"]))
    )

    lle_chart = lle_chart + alt.Chart(lle_new_data).mark_point(
        opacity=1, color="#CA03FF", filled=True, size=150).encode(
        alt.X("pca1"),
        alt.Y("pca2"),
        alt.ShapeValue('cross')).properties(**props).interactive()

    # Saving chart to file -- probably will be building this inside the app though...
    lle_chart.save("b41980/static/lle_spec.json")

    # Creating dictionary to pass into the html
    posts = {
        "livearea": h1.livearea,
        "bdrms": h1.bdrms,
        "baths": h1.baths,
        "one_story": h1.one_story,
        "att_garage": h1.att_garage,
        "basement": h1.basement,
        "prediction": pred
    }

    return render_template("predictresults.html", title="Prediction Results",
                           posts=posts)


@app.route('/resume')
def resume(id="/BrandonJenkinsResume.pdf"):
    return render_template('resume.html')
