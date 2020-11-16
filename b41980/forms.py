# Importing modules
from flask_wtf import FlaskForm
from wtforms import IntegerField, SubmitField
from wtforms.validators import DataRequired, NumberRange, InputRequired


class HouseForm(FlaskForm):
    livearea = IntegerField(label='Liveable Sqft',
                            validators=[DataRequired(),
                                        NumberRange(min=1, max=1000000)])
    bdrms = IntegerField(label='Bedrooms',
                         validators=[DataRequired(),
                                     NumberRange(min=1, max=1000)])
    baths = IntegerField(label='Baths',
                         validators=[DataRequired(),
                                     NumberRange(min=1, max=1000)])
    one_story = IntegerField(label='One Story Style',
                             validators=[InputRequired(),
                                         NumberRange(min=-1, max=1)])
    att_garage = IntegerField(label='Attached Garage',
                              validators=[InputRequired(),
                                          NumberRange(min=-1, max=1)])
    basement = IntegerField(label='Basement',
                            validators=[InputRequired(),
                                        NumberRange(min=-1, max=1)])
    predict = SubmitField(label="Predict")
