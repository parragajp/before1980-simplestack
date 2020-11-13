from flask import request, render_template, url_for, flash, redirect
from joblib import load
import numpy as np
from b41980.forms import HouseForm
from b41980.models import House
from b41980 import app, db

# Loading in our trained model
rf = load("rf_before1980.joblib")

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
        house = House(livearea=form.livearea.data, stories=form.stories.data,
                      bdrms=form.bdrms.data, baths=form.baths.data)
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
    new_instance = np.array([[h1.livearea, h1.stories, h1.bdrms, h1.baths]])

    # Predicting new_instance target
    prediction = rf.predict(new_instance)

    # Returning text depending on the prediction
    pred = "not" if prediction == 0 else ""

    # Creating dictionary to pass into the html
    posts = {
        "livearea": h1.livearea,
        "stories": h1.stories,
        "bdrms": h1.bdrms,
        "baths": h1.baths,
        "prediction": pred
    }

    return render_template("predictresults.html", title="Prediction Results",
                           posts=posts)
