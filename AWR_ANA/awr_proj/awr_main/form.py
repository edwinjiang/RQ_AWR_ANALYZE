#!/usr/bin/python
#-*- coding: UTF-8 -*-

from __future__ import unicode_literals
from flask_wtf import FlaskForm
from wtforms import RadioField, SubmitField, StringField, PasswordField
from wtforms.validators import DataRequired, Length

class upload_awr(FlaskForm):
    title = StringField('AWR 文件名',validators=[DataRequired(),Length(1,64)])
    submit = SubmitField('提交')

