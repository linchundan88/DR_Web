import os
import uuid
import pickle
import cv2
from my_module import my_dlp_helper, db_helper
import my_config
from my_module.my_dlp_helper import reload_my_config
from flask import Flask, request, render_template, session

app = Flask(__name__)
app.config['SECRET_KEY'] = '\xca\x5c\x86\x94\x98@\x02b\x1b7\x8c\x88]\x1b\xd7"+\xe6px@\xc3#\\'


@app.template_filter("my_round")
def do_round(num):
    return round(num, 2)

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == 'GET':
        lang = request.args.get('lang')
        session['lang'] = lang
        if lang == 'en':
            return render_template('login.html')
        else:
            return render_template('login_cn.html')

    if request.method == 'POST':
        lang = request.form['lang']
        username = request.form['username']
        password = request.form['password']

        login_OK = db_helper.login(username, password, write_log=True,
                                   source_ip=request.remote_addr)
        if login_OK:
            session['username'] = username
            session['lang'] = lang
            if lang == 'en':
                return render_template('uploadfile.html')
            else:
                return render_template('uploadfile_cn.html')
        else:
            if lang == 'en':
                error_msg = 'username or password error!'
                return render_template('login.html', error_msg=error_msg)
            else:
                error_msg = '账号或者密码错误!'
                return render_template('login_cn.html', error_msg=error_msg)

@app.route('/logout')
def logout():
    session.clear()
    return render_template('homepage.html')

@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == 'GET':
        if session.get('lang') is not None:
            if session.get('lang') == 'en':
                return render_template('register.html')

        return render_template('register_cn.html')

    if request.method == 'POST':
        lang = session.get('lang', 'en')

        email = request.form['email']
        name = request.form['name']
        tel = request.form['tel']
        company = request.form['company']
        title = request.form['title']

        if not '@' in email or email == '' or name == '' or company == '' or title == '':
            if lang == 'en':
                return render_template(request, 'register.html', error_msg='Please fill in the form correctly!')
            else:
                return render_template(request, 'register_cn.html', error_msg= '请按照规范填写表单')

        from my_module import db_helper
        results = db_helper.check_register(email)

        if len(results) > 0:  # 邮件账号已经使用
            if lang == 'en':
                return render_template('register.html', error_msg='This email has been used!')
            else:
                return render_template('register_cn.html', error_msg='该电子邮件已经被使用!')
        else:
            db_helper.do_register(email, name, tel, company, title)

            if lang == 'en':
                return render_template('register_ok.html')
            else:
                return render_template('register_ok_cn.html')

@app.route('/diagnose' , methods=["POST"])
def diagnose():
    if session.get('username') is None:
        return render_template('homepage.html')
    lang = str(session.get('lang'))

    file_upload = request.files['input_image_file']
    if file_upload.filename == '':
        if lang == 'en':
            return render_template('error.html')
        else:
            return render_template('error_cn.html')
    else:
        str_uuid = str(uuid.uuid1())
        session['image_uuid'] = str_uuid

        baseDir = os.path.dirname(os.path.abspath(__name__))
        img_dir = os.path.join(baseDir, 'static', 'imgs', str_uuid)
        os.makedirs(img_dir, exist_ok=True)
        filename = os.path.join(img_dir, file_upload.filename)
        file_upload.save(filename)

        filename_orig_new = os.path.join(img_dir, 'original.jpg')
        cv2.imwrite(filename_orig_new, cv2.imread(filename))

    reload_my_config()
    predict_result = my_dlp_helper.predict_all(filename, str_uuid, baseDir, lang,
             cam_type=my_config.CAM_TYPE, show_deepshap_dr=my_config.SHOW_DEEPSHAP_DR)

    pkl_save_file = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'predict_result.pkl')
    with open(pkl_save_file, 'wb') as f:
        pickle.dump(predict_result, f)

    username = session.get('username')
    diagnostic_results = predict_result['total_results']
    ip = request.remote_addr
    my_dlp_helper.save_to_db_diagnose(ip, username, str_uuid, diagnostic_results)

    if lang == 'en':
        file_view = 'diagnosis_result.html'
    else:
        file_view = 'diagnosis_result_cn.html'
    return render_template(file_view, realtime_diagnose=True, predict_result=predict_result)

@app.route('/view_diagnoses')
def view_diagnose_list():
    if session.get('username', None) is None:
        return render_template('homepage.html')
    username = session.get('username')

    from my_module import db_helper
    results = db_helper.diagnose_list(username)

    result_list = []
    for row in results:
        temp_list = [row[0], row[1], row[2], row[3], row[4]]
        result_list.append(temp_list)

    lang = session.get('lang', 'en')
    if lang == 'en':
        file_view = 'view_diagnoses_list.html'
    else:
        file_view = 'view_diagnoses_list_cn.html'
    return render_template(file_view, result_list =result_list)

@app.route('/view_diagnose_single')
def view_diagnose_single():
    if session.get('username', None) is None:
        return render_template('homepage.html')

    baseDir = os.path.dirname(os.path.abspath(__name__))

    str_uuid = request.args.get('uuid')
    pil_save_file = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'predict_result.pkl')
    pkl_file = open(pil_save_file, 'rb')
    predict_result = pickle.load(pkl_file)

    lang = str(session['lang'])
    if lang == 'en':
        file_view = 'diagnosis.html'
    else:
        file_view = 'diagnosis_cn.html'
    return render_template(file_view,  realtime_diagnose=False, predict_result= predict_result)


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=my_config.PORT,
        # debug=True
    )
