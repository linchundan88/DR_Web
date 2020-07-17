'''
#python3 MySQLdb
sudo apt-get install python-dev libmysqlclient-dev
sudo apt-get install python3-dev
pip install mysqlclient
'''

DB_TYPE = 'sqlite'  # mysql


import os
import sys
import sqlite3

'''
import json
import MySQLdb
def get_db_conn():
    json_file = os.path.join(os.path.abspath('.'), 'db_config.json')
    data = json.load(json_file)

    db = MySQLdb.connect(data['host'], data['username'], data['password'],
                         data['database'], use_unicode=True, charset='utf8')

    return db
'''

def get_db_conn():
    db_file = os.path.join(sys.path[0], 'database', 'dr.sqlite')
    conn = sqlite3.connect(db_file)
    return conn

def login(username, password, write_log=True, source_ip='127.0.0.1'):
    db = get_db_conn()
    cursor = db.cursor()

    from my_module.my_compute_digest import CalcSha1_str
    password_encrypt = CalcSha1_str(password)

    if DB_TYPE == 'mysql':
        sql = "SELECT * FROM tb_account WHERE username=%s and password_encrypt=%s and enabled=1"
    if DB_TYPE == 'sqlite':
        sql = "SELECT * FROM tb_account WHERE username=? and password_encrypt=? and enabled=1"
    cursor.execute(sql, (username, password_encrypt))
    results = cursor.fetchall()

    if len(results) == 1:
        if write_log:
            if DB_TYPE == 'mysql':
                sql = "insert into tb_log(username,log_memo) values(%s,%s)"
            if DB_TYPE == 'sqlite':
                sql = "insert into tb_log(username,log_memo) values(?,?)"
            cursor.execute(sql, (username, 'login_successful, ip:' + source_ip))
            db.commit()

        return True
    else:
        if write_log:
            if DB_TYPE == 'mysql':
                sql = "insert into tb_log(username,log_memo) values(%s,%s)"
            if DB_TYPE == 'sqlite':
                sql = "insert into tb_log(username,log_memo) values(?,?)"
            cursor.execute(sql, (username, 'login_failure, ip:' + source_ip))
            db.commit()

        return False

    cursor.close()
    db.close()


def check_register(email):
    db = get_db_conn()
    cursor = db.cursor()

    if DB_TYPE == 'mysql':
        sql_check_email = "SELECT * FROM tb_account WHERE username=%s and enabled=1"
    if DB_TYPE == 'sqlite':
        sql_check_email = "SELECT * FROM tb_account WHERE username=? and enabled=1"

    cursor.execute(sql_check_email, (email,))
    rs = cursor.fetchall()

    cursor.close()
    db.close()

    return rs

def do_register(email, name, tel, company, title):
    db = get_db_conn()
    cursor = db.cursor()
    if DB_TYPE == 'mysql':
        sql_insert = 'insert into tb_register(email, password, name, tel, company, title) values(%s, %s, %s, %s, %s, %s)'
    if DB_TYPE == 'sqlite':
        sql_insert = 'insert into tb_register(email, password, name, tel, company, title) values(?, ?, ?, ?, ?, ?)'
    cursor.execute(sql_insert, (email, '', name, tel, company, title))
    db.commit()
    db.close()

def diagnose_list(username):
    db = get_db_conn()
    cursor = db.cursor()

    if DB_TYPE == 'mysql':
        sql = 'select image_uuid,diagnostic_results,feedback_score,feedback_memo, DATE_FORMAT(date_time, "%%Y-%%m-%%d %%k:%%i:%%s") from tb_diagnoses WHERE username=%s order by date_time DESC '
    if DB_TYPE == 'sqlite':
        sql = 'select image_uuid,diagnostic_results,feedback_score,feedback_memo, date_time from tb_diagnoses WHERE username=? order by date_time DESC '

    cursor.execute(sql, (username,))
    results = cursor.fetchall()
    db.close()

    return results