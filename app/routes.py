from flask import render_template, flash, redirect, request, Flask
from app import app
from forms import ClassifyForm, SynonymForm
from Predict import predictKNN
from getSynonyms import sentenceVars, toString
from itertools import product
import nltk

application = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    user = {'username': 'Kate'}
    posts = [
        {
            'author': {'username': 'John'},
            'body': 'Beautiful day in Portland!'
        },
        {
            'author': {'username': 'Susan'},
            'body': 'The Avengers movie was so cool!'
        }
    ]
    return render_template('index.html', title='Home', user=user, posts=posts)

@app.route('/Classify', methods=['GET', 'POST'])
def Classify():
    form = ClassifyForm()
    if request.method =='POST':
        text = form.inputquestion.data
        response, confidence, outQuestion, similarity, final, classes, others = predictKNN(text)
        return render_template('Classify.html', title='Classify', form=form,
                               classification=outQuestion, similarity=similarity,
                               final=final, others=others)
    else:
        return render_template('Classify.html', title='Classify', form=form)
    #     if form.validate_on_submit():
    #     flash('You classified.."{}"'.format(
    #         form.inputquestion.data))
    #     return redirect('/index')
    # return render_template('Classify.html', title='Classify', form=form)


@app.route('/GetSynonyms', methods=['GET', 'POST'])
def GetSynonyms():
    form = SynonymForm()
    if request.method =='POST':
        text = form.inputquestion.data
        tokenize = nltk.word_tokenize(text)
        tags = nltk.pos_tag(tokenize)
        synonymSets = sentenceVars(tags)
        out1 = list(product(*synonymSets))
        out2 = toString(out1)
        return render_template('GetSynonyms.html', title='Get Synonyms', form=form, synonyms=out2)
    else:
        return render_template('GetSynonyms.html', title='Get Synonyms', form=form)
    # if form.validate_on_submit():
    #     flash('You got synonyms for.."{}"'.format(
    #         form.inputquestion.data))
    #     return redirect('/index')
    # return render_template('GetSynonyms.html', title='GetSynonyms', form=form)
if __name__ == "__main__":
    applicaiton.run(host='0.0.0.0')
