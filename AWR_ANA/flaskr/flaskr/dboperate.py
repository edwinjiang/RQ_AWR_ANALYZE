#!/usr/bin/python
#encoding:utf8

import pymysql as db
from contextlib import contextmanager

@contextmanager
def get_conn(**kargs):
    conn = db.connect(host=kargs.get('host','localhost'),
                      user=kargs.get('user'),
                      passwd=kargs.get('passwd'),
                      port=kargs.get('port', 3306),
                      db=kargs.get('db'))
    try:
        yield conn
    finally:
        conn.close()


def execute_sql(conn, sql):
    with conn as cur:
        for loc in range(len(sql)):
            cur.execute(sql[loc])



