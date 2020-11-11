# Importing modules
from flask_wtf import FlaskForm
from wtforms import IntegerField, SubmitField
from wtforms.validators import DataRequired, NumberRange


class HouseForm(FlaskForm):
    livearea = IntegerField(label='Liveable Sqft',
                            validators=[DataRequired(),
                                        NumberRange(min=1, max=1000000)])
    stories = IntegerField(label='Stories',
                           validators=[DataRequired(),
                                       NumberRange(min=1, max=1000)])
    bdrms = IntegerField(label='Bedrooms',
                         validators=[DataRequired(),
                                     NumberRange(min=1, max=1000)])
    baths = IntegerField(label='Baths',
                         validators=[DataRequired(),
                                     NumberRange(min=1, max=1000)])
    predict = SubmitField(label="Predict")
