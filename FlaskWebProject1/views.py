"""
Routes and views for the flask application.
"""
# -*- coding: utf-8 -*-
from datetime import datetime
from flask import render_template,request,Flask
from FlaskWebProject1 import app
import FlaskWebProject1.finalversion2 as fv
import requests
import cv2
@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year,
    )

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact',
        year=datetime.now().year,
        message='Feel free to contact with us.'
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='About the Fresh Keeper'
    )

@app.route('/Main')
def Main():
    return render_template(
        'MainPage.html',
        title="MainPage",
        )

@app.route('/Main/Tool')
def Tool():
    return render_template(
        'ToolPage.html',
        title="ToolPage",
        )

@app.route('/Main/Display',methods=['POST','GET'])
def Display():
    img_name=""
    if request.method=="POST":
        ori_img=request.files.get('ori_img')
        file_name=ori_img.filename
        pre="./FlaskWebProject1/static/"
        img_path=pre+f"{file_name}"
        if ori_img:
            ori_img.save(img_path)
            print("OK")
            resp=Analyze(img_path)
            res_img=resp[0]
            img_name=file_name.split(".")[0]+"_res."+file_name.split(".")[-1]
            res_img_path=pre+img_name
            cv2.imwrite(res_img_path,res_img)
            return render_template(
                'Display.html',
                title="DisplayPage",
                image_name=img_name
            )
    return render_template(
        'Display.html',
        title="DisplayPage",
        image_name=img_name
        )

@app.route('/Main/Introduction')
def Introduction():
    return render_template(
        'Introduction.html',
        title="introducePage",
        )

@app.route('/Main/MoreInf')
def MoreInf():
    return render_template(
        'MoreInf.html',
        title="More Information",
        )


@app.route('/Main/ChooseFunc')
def ChooseFunc():
    return render_template(
        'ChooseFunc.html',
        title="Choose the Function",
        )


@app.route('/Main/Knowledge')
def Knowledge():
    return render_template(
        'Knowledge.html',
        title="Knowledge of the fruit",
        )

@app.route('/Main/FruitCommunity')
def FruitCommunity():
    return render_template(
        'FruitCommunity.html',
        title="The community for the share of food",
        )

def Analyze(ori_img_path):
    results=fv.main1(ori_img_path)
    return results