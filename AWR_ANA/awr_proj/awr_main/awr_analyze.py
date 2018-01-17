#!/usr/bin/bash
#encoding:utf8

from hdfs.client import Client

client = Client("http://192.168.40.4:50070")

with client.read('/awrreport/test.log') as fs:
    content = fs.read()
    print content