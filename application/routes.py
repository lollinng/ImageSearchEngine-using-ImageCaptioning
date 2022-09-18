import os
from flask import render_template, redirect, flash, request
from application import app

from application.output import run_query


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['query'].lower()
        # "White dogs running"
        values = run_query(text)
        return render_template('home.html',values=values)

    return render_template('home.html')


    
