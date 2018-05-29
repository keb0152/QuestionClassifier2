from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

class ClassifyForm(FlaskForm):
    inputquestion = StringField('What is your Question?', validators=[DataRequired()])
    submit = SubmitField('Classify')

class SynonymForm(FlaskForm):
    inputquestion = StringField('What is your Question?', validators=[DataRequired()])
    submit = SubmitField('Get Synonyms')