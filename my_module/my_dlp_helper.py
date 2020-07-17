import os
import xmlrpc.client
import json
import shutil
import cv2
from my_module import my_preprocess
import my_config

def get_disease_name(no, class_type='Stage', lang='en'):
    if class_type == 'Gradable':
        json_file = 'classid_to_human_Gradable.json'
    elif class_type == 'Left_Right':
        json_file = 'classid_to_human_Left_Right.json'
    elif class_type == 'DR':
        json_file = 'classid_to_human_DR.json'

    else:
        raise Exception('get disease name error!')

    if lang == 'cn':
        json_file = 'cn_' + json_file

    baseDir = os.path.dirname(os.path.abspath(__name__))
    json_file = os.path.join(os.path.join(baseDir, 'diseases_json'), json_file)

    with open(json_file, 'r') as json_file:
        data = json.load(json_file)
        for i in range(len(data['diseases'])):
            if data['diseases'][i]['NO'] == no:
                return data['diseases'][i]['NAME']

def predict_single_class(img_source, class_type='Stage',
                         softmax_or_multilabels='softmax'):
    if class_type == 'Gradable':  # image quality
        SERVER_URL = my_config.URL_GRADABLE
    elif class_type == 'Left_Right':
        SERVER_URL = my_config.URL_LEFT_RIGHT
    elif class_type == 'DR':
        SERVER_URL = my_config.URL_DR

    with xmlrpc.client.ServerProxy(SERVER_URL) as proxy1:
        if softmax_or_multilabels == 'softmax':
            prob_list, pred_list, prob_total, pred_total, correct_model_no = proxy1.predict_softmax(
                img_source)
        else:
            prob_list, pred_list, prob_total, pred_total, correct_model_no = proxy1.predict_multi_labels(
                img_source)
        return prob_list, pred_list, prob_total, pred_total, correct_model_no

def predict_all(file_img_source, str_uuid, baseDir, lang,
                cam_type='CAM', show_deepshap_dr=False):
    img_source = cv2.imread(file_img_source)
    predict_result = {}

    #region preprocess
    img_file_resized_384 = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'resized_384.jpg')
    cv2.imwrite(img_file_resized_384, cv2.resize(img_source, (384, 384)))
    predict_result['img_file_resized_384'] = img_file_resized_384.replace(baseDir, '')

    img_file_preprocessed_384 = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'preprocessed_384.jpg')
    my_preprocess.do_preprocess(img_source,  crop_size=384,
                img_file_dest=img_file_preprocessed_384, add_black_pixel_ratio=0.02)
    predict_result['img_file_preprocessed_384'] = img_file_preprocessed_384.replace(baseDir, '')
    #endregion

    prob_list_gradable, pred_list_gradable, prob_total_gradable, pred_total_gradable, correct_model_no_gradable = \
            predict_single_class(img_file_preprocessed_384, class_type='Gradable', softmax_or_multilabels='softmax')
    predict_result['img_gradable'] = pred_total_gradable
    predict_result["img_gradable_0_name"] = get_disease_name(0, class_type='Gradable', lang=lang)
    predict_result["img_gradable_0_prob"] = round(prob_total_gradable[0], 3) * 100
    predict_result["img_gradable_1_name"] = get_disease_name(1, class_type='Gradable', lang=lang)
    predict_result["img_gradable_1_prob"] = round(prob_total_gradable[1], 3) * 100

    if predict_result['img_gradable'] == 1: #image quality ungradable
        if my_config.UNGRADABLE_RESHOOTING:
            if lang == 'en':
                predict_result['total_results'] = 'Rephotograph'
            else:
                predict_result['total_results'] = '诊断结果是:图片质量不可评级，建议重拍。'

            predict_result['recommended'] = predict_result['total_results']
            return predict_result

    if my_config.ENABLE_LEFT_RIGHT:
        prob_list_left_right, pred_list_left_right, prob_total_left_right, pred_total_left_right, correct_model_no_left_right = \
                predict_single_class(img_file_preprocessed_384, class_type='Left_Right', softmax_or_multilabels='softmax')
        predict_result['img_left_right'] = pred_total_left_right
        predict_result["img_left_right_0_name"] = get_disease_name(0, class_type='Left_Right', lang=lang)
        predict_result["img_left_right_0_prob"] = round(prob_total_left_right[0], 3) * 100
        predict_result["img_left_right_1_name"] = get_disease_name(1, class_type='Left_Right', lang=lang)
        predict_result["img_left_right_1_prob"] = round(prob_total_left_right[1], 3) * 100

    prob_list_dr, pred_list_dr, prob_total_dr, pred_total_dr, correct_model_no_dr = \
        predict_single_class(img_file_preprocessed_384, class_type='DR', softmax_or_multilabels='softmax')
    predict_result['img_dr'] = pred_total_dr
    predict_result["img_dr_0_name"] = get_disease_name(0, class_type='DR', lang=lang)
    predict_result["img_dr_0_prob"] = round(prob_total_dr[0], 3) * 100
    predict_result["img_dr_1_name"] = get_disease_name(1, class_type='DR', lang=lang)
    predict_result["img_dr_1_prob"] = round(prob_total_dr[1], 3) * 100

    if cam_type != '' and predict_result['img_dr'] == 1:
        if cam_type == 'CAM':
            cam_port = my_config.URL_CAM_DR
            cam_relu = my_config.CAM_RELU
        if cam_type == 'grad_cam':
            cam_port = my_config.URL_grad_cam_DR
        if cam_type == 'gradcam_plus':
            cam_port = my_config.URL_gradcam_plus_DR

        with xmlrpc.client.ServerProxy(cam_port) as proxy1:
            model_no = 0
            file_cam_tmp = proxy1.server_cam(model_no,
                    img_file_preprocessed_384, predict_result['img_dr'],
                    cam_relu, my_config.BLEND_ORIGINAL_IMAGE)
            if my_config.BLEND_ORIGINAL_IMAGE:
                file_ext = '.gif'
            else:
                file_ext = '.jpg'
            filename_heatmap = os.path.join(baseDir, 'static', 'imgs', str_uuid, cam_type + file_ext)
            shutil.copy(file_cam_tmp, filename_heatmap)
            predict_result["Heatmap_cam_dr"] = filename_heatmap.replace(baseDir, '')

    if show_deepshap_dr and predict_result['img_dr'] == 1:
        with xmlrpc.client.ServerProxy(my_config.URL_DEEPSHAP_DR) as proxy1:
            model_no = 0
            list_classes, list_images = proxy1.server_shap_deep_explainer(model_no,
                    img_file_preprocessed_384, 1, my_config.BLEND_ORIGINAL_IMAGE)
            if my_config.BLEND_ORIGINAL_IMAGE:
                file_ext = '.gif'
            else:
                file_ext = '.jpg'
            filename_heatmap = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'Heatmap_deepshap_stage_1' + file_ext)
            shutil.copy(list_images[0], filename_heatmap)
            predict_result["Heatmap_deepshap_dr"] = filename_heatmap.replace(baseDir, '')

    #region disease name
    if lang == 'en':
        predict_result['total_results'] = ''
    else:
        predict_result['total_results'] = '建议 : '

    referal = False
    if (predict_result['img_dr'] == 1):
        referal = True
        if lang == 'en':
            predict_result['total_results'] += 'Referral-warranted'
        else:
            predict_result['total_results'] += '转诊'

    if predict_result['img_gradable'] == 1:
        predict_result['total_results'] = 'Rephotograph'

    predict_result['detected'] = ''

    if referal:
        if lang == 'en':
            disease_name = ' Detected positive Diabetic Retinopathy-related feature: '
        else:
            disease_name = '检测到 : '
        if predict_result['img_dr'] == 1:
            disease_name += get_disease_name(predict_result['img_dr'],
                                             class_type='DR', lang=lang)
            disease_name += ', '

        disease_name = disease_name[:-2]
        disease_name += ''
        predict_result['detected'] = disease_name

    predict_result['recommended'] = predict_result['total_results']
    predict_result['total_results'] = predict_result['detected'] + '\n' + predict_result['total_results']

    #endregion

    return predict_result


def save_to_db_diagnose(ip, username, image_uuid, diagnostic_results, del_duplicate=False):
    from my_module import db_helper
    db = db_helper.get_db_conn()
    cursor = db.cursor()

    if del_duplicate:
        if db_helper.DB_TYPE == 'mysql':
            sql = "delete from tb_diagnoses where image_uuid = %s"
        if db_helper.DB_TYPE == 'sqlite':
            sql = "delete from tb_diagnoses where image_uuid = ?"
        cursor.execute(sql, (image_uuid,))
        db.commit()

    if db_helper.DB_TYPE == 'mysql':
        sql = "insert into tb_diagnoses(IP,username, image_uuid, diagnostic_results) values(%s,%s,%s,%s) "
    if db_helper.DB_TYPE == 'sqlite':
        sql = "insert into tb_diagnoses(IP,username, image_uuid, diagnostic_results) values(?,?,?,?) "
    cursor.execute(sql, (ip, username, image_uuid, diagnostic_results))
    db.commit()

    db.close()


def reload_my_config():
    from imp import reload
    reload(my_config)