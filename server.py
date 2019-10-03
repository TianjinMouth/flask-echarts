import random
from pyecharts import Grid, Bar, Line, Overlap, Timeline, Map, Pie, Boxplot
from flask import Flask, render_template
from pyecharts import Page
from sklearn import preprocessing
from monitor_v1 import *
from BinningPSI import TableBinPsi
import pandas as pd
import datetime
from sklearn.metrics import roc_auc_score
# from pyecharts.engine import ECHAERTS_TEMPLATE_FUNCTIONS
# from pyecharts.conf import PyEchartsConfig
import warnings
# from flask.templating import Environment
warnings.filterwarnings('ignore')

# # ----- Adapter ---------
# class FlaskEchartsEnvironment(Environment):
#     def __init__(self, *args, **kwargs):
#         super(FlaskEchartsEnvironment, self).__init__(*args, **kwargs)
#         self.pyecharts_config = PyEchartsConfig(jshost='/static/js')
#         self.globals.update(ECHAERTS_TEMPLATE_FUNCTIONS)
#
#
# # ---User Code ----
#
# class MyFlask(Flask):
#     jinja_environment = FlaskEchartsEnvironment
#
#
# app = MyFlask(__name__)


def features_split_time(tdf, time, time_space, flag):
    cut_col = 'create_time'
    df = tdf.copy()
    temp_import_data = pd.DataFrame()
    if time_space == '-1':
        import_data = df[(df[cut_col] >= time[0]) & (df[cut_col] <= time[1])]
        import_data['time_flag'] = flag
        temp_import_data = pd.concat([temp_import_data, import_data])
    else:
        import_time_list = [str(t)[:10] for t in
                            pd.date_range(start=time[0], end=time[1], freq=time_space + 'D', name='dt')]
        if time[1] not in import_time_list:
            import_time_list.append(time[1])
        for i in range(len(import_time_list) - 1):
            import_data = df[(df[cut_col] >= import_time_list[i]) & (df[cut_col] <= import_time_list[i + 1])]
            import_data['time_flag'] = flag
            temp_import_data = pd.concat([temp_import_data, import_data])
    return temp_import_data


def get_app_count(x):
    if str(x) == 'nan':
        return 0
    else:
        return len(set(eval(x)))


def get_gender(x):
    if str(x) == 'male':
        return 1
    elif str(x) == 'female':
        return 0


app = Flask(__name__)
# REMOTE_HOST = "http://127.0.0.1:1234/"


@app.route("/")
def report_chart():
    # define a page
    page = Page()
    bar = Bar(subtitle='本监控页面往期数据未涉及到还款表现的图表均选自进件时间在 '+old_period.split(',')[0]+' 至 '+old_period.split(',')[1]+' 之间 ' +
                       '\n近期数据选自进件时间在 '+new_period.split(',')[0]+' 至 '+new_period.split(',')[1]+' 之间 ' +
                       '\n然涉及到还款表现的往期均选自应还日期在 '+old_plan_repay_date+' 至 '+new_plan_repay_date+' 之间 '+', 近期选自近 '+str(td)+' 天内',
                       title_pos='left', subtitle_text_size=16, height=80, width=1200)
    page.add(bar)
    for i in list(set(['age', 'gender', 'app_count', 'phonebook_contacts_nums', 'history_month_cnt_180_sum',
                       'gsm_communication_way_dail_180_sum', 'gsm_communication_way_dailed_180_sum'] + abnormal_list)):
        grid = Grid(width=1200)
        attr = list(tmp[tmp['var'] == i]['bins'])
        bar = Bar(i, "重要|可能存在问题的字段"+"  即psi大于"+str(abnormal_threshold))
        bar.add('近期', attr, list(tmp[tmp['var'] == i]['value'].iloc[:, 0]),
                is_label_show=False, is_datazoom_show=True, xaxis_interval=0, datazoom_range=[0,100], xaxis_rotate=20,
                yaxis_rotate=20)
        bar.add('往期', attr, list(tmp[tmp['var'] == i]['value'].iloc[:, 1]),
                is_label_show=False, is_datazoom_show=True,  xaxis_interval=0, datazoom_range=[0,100], xaxis_rotate=20,
                yaxis_rotate=20)
        boxplot = Boxplot()
        x_axis = ['近期', '往期']
        y_axis = [
            list(current_import_data[i].fillna(current_import_data[i].mean()).values),
            list(old_import_data[i].fillna(old_import_data[i].mean()).values)
        ]
        _yaxis = boxplot.prepare_data(y_axis)  # 转换数据
        boxplot.add("", x_axis, _yaxis, yaxis_max=_yaxis[0][3] + 1.5 * (_yaxis[0][3] - _yaxis[0][1]))
        grid.add(bar, grid_right="60%")
        grid.add(boxplot, grid_left="60%")

        page.add(grid)
        # page.add(bar)

    # timeline = Timeline(is_auto_play=True, timeline_bottom=0, width=1200)
    # for i in list(set(important_features) - {'gender'}):
    #     boxplot = Boxplot(i, '重要特征箱线图,分新老户和渠道', title_text_size=14)
    #     for j in ['ff', 'xianjincard', 'ygx']:
    #         for k in ['old', 'new']:
    #             a = start_point[(start_point['inner_sources'] == j) & (start_point['user_tag'] == k)][i]
    #             x_axis = [i]
    #             y_axis = [
    #                 list(a.fillna(a.mean()).values)
    #             ]
    #             qua = boxplot.prepare_data(y_axis)
    #             q3 = qua[0][3]
    #             q1 = qua[0][1]
    #             boxplot.add(j + ' + ' + k, x_axis, qua, is_xaxis_show=False, yaxis_max=q3 + 1.5 * (q3 - q1))
    #     timeline.add(boxplot, i)
    # page.add(timeline)

    for le in range(0, len(list(set(important_features) - {'gender'})), 2):
        i = list(set(important_features) - {'gender'})[le]
        q = list(set(important_features) - {'gender'})[le + 1]

        boxplot = Boxplot()
        qua = boxplot.prepare_data([list(start_point[i].fillna(start_point[i].mean()).values)])
        q3 = qua[0][3]
        q1 = qua[0][1]

        grid = Grid(width=1200)
        boxplot = Boxplot(i + '&' + q, '重要特征箱线图,分新老户和渠道', title_text_size=12)
        for j in ['ff', 'xianjincard', 'ygx']:
            for k in ['old', 'new']:
                a = start_point[(start_point['inner_sources'] == j) & (start_point['user_tag'] == k)][i]
                x_axis = [i]
                y_axis = [
                    list(a.fillna(a.mean()).values)
                ]
                boxplot.add(j + ' + ' + k, x_axis, boxplot.prepare_data(y_axis), is_xaxis_show=False,
                            yaxis_max=q3 + 1.5 * (q3 - q1), yaxis_min=min(y_axis[0]))
        grid.add(boxplot, grid_left="60%")

        boxplot = Boxplot()
        qua = boxplot.prepare_data([list(start_point[q].fillna(start_point[q].mean()).values)])
        q3 = qua[0][3]
        q1 = qua[0][1]

        boxplot = Boxplot('', '', title_text_size=12)
        for j in ['ff', 'xianjincard', 'ygx']:
            for k in ['old', 'new']:
                a = start_point[(start_point['inner_sources'] == j) & (start_point['user_tag'] == k)][q]
                x_axis = [q]
                y_axis = [
                    list(a.fillna(a.mean()).values)
                ]
                boxplot.add(j + ' + ' + k, x_axis, boxplot.prepare_data(y_axis), is_xaxis_show=False,
                            yaxis_max=q3 + 1.5 * (q3 - q1), yaxis_min=min(y_axis[0]))
        grid.add(boxplot, grid_right="60%")
        page.add(grid)

    old_null = old_import_data[important_features].isnull().sum() / old_import_data.shape[0]
    new_null = current_import_data[important_features].isnull().sum() / current_import_data.shape[0]

    bar = Bar('空值分布', "只限于重要特征", width=1200)
    bar.add('近期', important_features, new_null.values,
            is_label_show=False, is_datazoom_show=True, datazoom_range=[10, 30], datazoom_type="both", xaxis_interval=0)
    bar.add('往期', important_features, old_null.values,
            is_label_show=False, is_datazoom_show=True, datazoom_range=[10, 30], datazoom_type="both", xaxis_interval=0)
    page.add(bar)

    for i in ['education', 'month_income', 'home_type']:
        attr1 = list(raw[(raw['inner_sources'] == 'ff') & (raw['plan_repay_date'] >= old_plan_repay_date)].groupby(raw[(raw['inner_sources'] == 'ff') & (raw['plan_repay_date'] >= old_plan_repay_date)][i].values)[i].agg(
            'count').index)
        attr1 = [i.replace('Between', '').replace('And', '-').replace('Up', '') for i in attr1]
        v1 = list(raw[(raw['inner_sources'] == 'ff') & (raw['plan_repay_date'] >= old_plan_repay_date)].groupby(raw[(raw['inner_sources'] == 'ff') & (raw['plan_repay_date'] >= old_plan_repay_date)][i].values)[i].agg(
            'count').values)

        attr2 = list(
            raw[(raw['inner_sources'] == 'xianjincard') & (raw['plan_repay_date'] >= old_plan_repay_date)].groupby(raw[(raw['inner_sources'] == 'xianjincard') & (raw['plan_repay_date'] >= old_plan_repay_date)][i].values)[
                i].agg('count').index)
        attr2 = [i.replace('Between', '').replace('And', '-').replace('Up', '') for i in attr2]
        v2 = list(
            raw[(raw['inner_sources'] == 'xianjincard') & (raw['plan_repay_date'] >= old_plan_repay_date)].groupby(raw[(raw['inner_sources'] == 'xianjincard') & (raw['plan_repay_date'] >= old_plan_repay_date)][i].values)[
                i].agg('count').values)

        attr3 = list(
            raw[(raw['inner_sources'] == 'ygx') & (raw['plan_repay_date'] >= old_plan_repay_date)].groupby(raw[(raw['inner_sources'] == 'ygx') & (raw['plan_repay_date'] >= old_plan_repay_date)][i].values)[
                i].agg('count').index)
        attr3 = [i.replace('Between', '').replace('And', '-').replace('Up', '') for i in attr3]
        v3 = list(
            raw[(raw['inner_sources'] == 'ygx') & (raw['plan_repay_date'] >= old_plan_repay_date)].groupby(raw[(raw['inner_sources'] == 'ygx') & (raw['plan_repay_date'] >= old_plan_repay_date)][i].values)[
                i].agg('count').values)

        pie = Pie(i, title_pos='center', width=1200)
        pie.add(
            "ff",
            attr1,
            v1,
            center=[17, 50],
            is_random=True,
            radius=[30, 75],
            legend_text_size=8,
            label_text_color=None,
            label_text_size=8,
            is_label_show=True,
            legend_orient="vertical",
            # rosetype="area",
            legend_pos="left",
        )
        pie.add(
            "xianjincard",
            attr2,
            v2,
            center=[51, 50],
            is_random=True,
            radius=[30, 75],
            legend_text_size=8,
            label_text_color=None,
            label_text_size=8,
            legend_orient="vertical",
            # rosetype="area",
            legend_pos="left",
        )
        pie.add(
            "ygx",
            attr3,
            v3,
            center=[85, 50],
            is_random=True,
            radius=[30, 75],
            legend_text_size=8,
            label_text_color=None,
            label_text_size=8,
            is_label_show=True,
            legend_orient="vertical",
            # rosetype="area",
            legend_pos="left",
        )
        page.add(pie)

    value = list(old_import_data.groupby(old_import_data['ipprovince'].values)['ipprovince'].agg('count').values)
    attr = list(old_import_data.groupby(old_import_data['ipprovince'].values)['ipprovince'].agg('count').index)

    # attr_ = pd.DataFrame(value,attr).reset_index().groupby('index').agg({'sum'}).reset_index()['index']
    # attr_ = [i.replace('未知', '南海诸岛') for i in attr_]
    # value_ = pd.DataFrame(value,attr).reset_index().groupby('index').agg({'sum'}).reset_index()[0]['sum']/
    # sum(pd.DataFrame(value,attr).reset_index().groupby('index').agg({'sum'}).reset_index()[0]['sum'])
    #
    # min_max_scaler = preprocessing.MinMaxScaler()
    # value_minmax = min_max_scaler.fit_transform(np.array(value_).reshape(-1, 1))
    map = Map("ip_province", width=1200, height=600)
    map.add(
        "往期",
        attr,
        value,
        maptype="china",
        is_visualmap=True,
        is_map_symbol_show=False,
        is_label_show=True,
        is_roam=False,
        visual_text_color="#000",
    )

    page.add(map)

    value = list(
        current_import_data.groupby(current_import_data['ipprovince'].values)['ipprovince'].agg('count').values)
    attr = list(current_import_data.groupby(current_import_data['ipprovince'].values)['ipprovince'].agg('count').index)

    # attr_ = pd.DataFrame(value,attr).reset_index().groupby('index').agg({'sum'}).reset_index()['index']
    # attr_ = [i.replace('未知', '南海诸岛') for i in attr_]
    # value_ = pd.DataFrame(value,attr).reset_index().groupby('index').agg({'sum'}).reset_index()[0]['sum']/
    # sum(pd.DataFrame(value,attr).reset_index().groupby('index').agg({'sum'}).reset_index()[0]['sum'])
    #
    # min_max_scaler = preprocessing.MinMaxScaler()
    # value_minmax = min_max_scaler.fit_transform(np.array(value_).reshape(-1, 1))
    map = Map("ip_province", width=1200, height=600)
    map.add(
        "近期",
        attr,
        value,
        maptype="china",
        is_visualmap=True,
        is_map_symbol_show=False,
        is_label_show=True,
        is_roam=False,
        visual_text_color="#000",
    )

    page.add(map)

    value = list(old_import_data.groupby(old_import_data['gps_province'].values)['gps_province'].agg('count').values)
    attr = [i[0:2] if i[0:2] != '内蒙' else '内蒙古' for i in
            list(old_import_data.groupby(old_import_data['gps_province'].values)['gps_province'].agg('count').index)]

    attr_ = pd.DataFrame(value, attr).reset_index().groupby('index').agg({'sum'}).reset_index()['index']
    attr_ = [i.replace('未知', '南海诸岛') for i in attr_]
    value_ = pd.DataFrame(value, attr).reset_index().groupby('index').agg({'sum'}).reset_index()[0]['sum']/sum(pd.DataFrame(value, attr).reset_index().groupby('index').agg({'sum'}).reset_index()[0]['sum'])

    min_max_scaler = preprocessing.MinMaxScaler()
    value_minmax = min_max_scaler.fit_transform(np.array(value_).reshape(-1, 1))

    map = Map("gps_province", width=1200, height=600)
    map.add(
        "往期",
        attr_,
        np.round(value_minmax * 100, 1),
        maptype="china",
        is_visualmap=True,
        is_map_symbol_show=False,
        is_label_show=True,
        visual_text_color="#000",
        is_roam=False,
        is_piecewise=True,
        pieces=[
            {"max": 100, "min": 80, "label": "80-100"},
            {"max": 80, "min": 40, "label": "40-80"},
            {"max": 40, "min": 30, "label": "20-40"},
            {"max": 30, "min": 20, "label": "20-30"},
            {"max": 20, "min": 10, "label": "10-20"},
            {"max": 10, "min": 0, "label": "0-10"},
        ]
    )
    page.add(map)

    value = list(
        current_import_data.groupby(current_import_data['gps_province'].values)['gps_province'].agg('count').values)
    attr = [i[0:2] if i[0:2] != '内蒙' else '内蒙古' for i in list(
        current_import_data.groupby(current_import_data['gps_province'].values)['gps_province'].agg('count').index)]

    attr_ = pd.DataFrame(value, attr).reset_index().groupby('index').agg({'sum'}).reset_index()['index']
    attr_ = [i.replace('未知', '南海诸岛') for i in attr_]
    value_ = pd.DataFrame(value, attr).reset_index().groupby('index').agg({'sum'}).reset_index()[0]['sum'] / sum(
        pd.DataFrame(value, attr).reset_index().groupby('index').agg({'sum'}).reset_index()[0]['sum'])

    min_max_scaler = preprocessing.MinMaxScaler()
    value_minmax = min_max_scaler.fit_transform(np.array(value_).reshape(-1, 1))

    map = Map("gps_province", width=1200, height=600)
    map.add(
        "近期",
        attr_,
        np.round(value_minmax * 100, 1),
        maptype="china",
        is_visualmap=True,
        is_map_symbol_show=False,
        is_label_show=True,
        visual_text_color="#000",
        is_roam=False,
        is_piecewise=True,
        pieces=[
            {"max": 100, "min": 80, "label": "80-100"},
            {"max": 80, "min": 40, "label": "40-80"},
            {"max": 40, "min": 20, "label": "20-40"},
            {"max": 30, "min": 20, "label": "20-30"},
            {"max": 20, "min": 10, "label": "10-20"},
            {"max": 10, "min": 0, "label": "0-10"},
        ]
    )
    page.add(map)
    grid = Grid(width=1200)

    a = list(oo.groupby('date')['operator_model_pred'].count().index)
    b = list(oo.groupby('date')['operator_model_pred'].count().values)
    bar = Bar('loan_population', "还款人数", width=1200)
    bar.add('', a, b, is_label_show=False, is_datazoom_show=True, is_random=True, datazoom_range=[0,100])
    page.add(bar)

    a = list(oo.groupby('date')['operator_model_pred'].mean().index)
    b = list(oo.groupby('date')['operator_model_pred'].mean().values)
    c = list(oo.groupby('date')['is_overdue'].mean().index)
    d = list(oo.groupby('date')['is_overdue'].mean().values)
    bar = Bar('opmodel_score', "运营商模型分数", width=1200)
    bar.add('', a, b, is_label_show=False, is_datazoom_show=True, is_random=True, datazoom_range=[0,100])

    line = Line(width=1200)
    line.add("还款率", c, d)
    overlap = Overlap(width=1200, height=600)
    overlap.add(bar)
    overlap.add(line, is_add_yaxis=True, yaxis_index=1)

    grid.add(overlap, grid_right="20%")
    page.add(grid)

    a = list(df_.groupby('modeltype')['is_overdue_'].mean().index)
    b = list(df_.groupby('modeltype')['is_overdue_'].mean().values)
    bar = Bar('model_type', "模型分分析", width=1200)
    bar.add('', a, b, is_label_show=False, is_datazoom_show=True, is_random=True, xaxis_interval=0, datazoom_range=[0,100])
    page.add(bar)

    old_a = list(old.groupby('modeltype')['order_id'].count().index)
    old_b = list(old.groupby('modeltype')['order_id'].count().values)
    new_a = list(new.groupby('modeltype')['order_id'].count().index)
    new_b = list(new.groupby('modeltype')['order_id'].count().values)
    bar = Bar('', "", width=1200)
    bar.add('往期', old_a, old_b, is_label_show=False, is_datazoom_show=True, is_random=True, xaxis_interval=0,
            datazoom_range=[0,100])
    bar.add('近期', new_a, new_b, is_label_show=False, is_datazoom_show=True, is_random=True, xaxis_interval=0,
            datazoom_range=[0,100])
    page.add(bar)

    timeline = Timeline(is_auto_play=True, timeline_bottom=0)
    attr = ['往期', '近期']
    for i in list(set(df_.modeltype)):
        a = list()
        a.append(old[old['modeltype'] == i]['is_overdue'].mean())
        a.append(new[new['modeltype'] == i]['is_overdue'].mean())
        bar = Bar(i, "")
        bar.add(i, attr, a, is_label_show=False, is_random=True)

        timeline.add(bar, i)

    page.add(timeline)

    for i in ['operator_model_pred', 'precredit_score', 'third_model_pred']:
        line = Line(i, '近期往期对比', width=1200)

        _old = old[old[i] < 99].reset_index(drop=True)
        _new = new[new[i] < 99].reset_index(drop=True)

        tmp_1 = list(_old[i])
        tmp_2 = list(_new[i])
        attr = [i / 100 for i in list(range(0, 100, 1))]
        tt_1 = [(len([x for x in tmp_1 if x >= attr[i] and x < attr[i + 1]]) / len(tmp_1)) if len(tmp_1) != 0 else 0 for i in list(range(0, 100))]
        tt_2 = [(len([x for x in tmp_2 if x >= attr[i] and x < attr[i + 1]]) / len(tmp_2)) if len(tmp_2) != 0 else 0 for i in list(range(0, 100))]

        line.add("往期", [str(x) for x in attr], tt_1,
                 is_smooth=True, is_label_show=False, mark_line=["max", "average"], is_fill=True,
                 line_opacity=0.2,
                 area_opacity=0.4,
                 is_random=True,
                 is_symbol_show=False)
        line.add("近期", [str(x) for x in attr], tt_2,
                 is_smooth=True, is_label_show=False, mark_line=["max", "average"], is_fill=True,
                 line_opacity=0.2,
                 area_opacity=0.4,
                 # area_color="#000",
                 is_random=True,
                 is_symbol_show=False)

        page.add(line)
        line = Line('', '新老用户渠道对比', width=1200)
        for j in ['ff', 'xianjincard', 'ygx']:
            for k in list(set(df_['user_tag'])):
                _channel = df_[(df_['inner_sources'] == j) & (df_['user_tag'] == k) & (df_[i] < 99) & (df_['plan_repay_date'] >= old_plan_repay_date)].reset_index(drop=True)
                _channel_list = list(_channel[i])
                attr = [i / 100 for i in list(range(0, 100, 1))]
                _channel_value = [(len([x for x in _channel_list if x >= attr[i] and x < attr[i + 1]]) / len(_channel_list)) if len(_channel_list) != 0 else 0 for i in list(range(0, 100))]
                line.add(j + ' + ' + k, [str(x) for x in attr], _channel_value,
                         is_smooth=True, is_label_show=False, mark_line=["max", "average"], is_fill=True,
                         line_opacity=0.2,
                         area_opacity=0.4,
                         # is_random=True,
                         is_symbol_show=False)
        page.add(line)

    for i in ['operator_model_pred', 'third_model_pred', 'precredit_score']:
        line = Line(i, 'Roc_auc', width=1200)
        for j in ['ff', 'xianjincard', 'ygx']:
            _auc = df_[(df_['inner_sources'] == j) & (df_[i] < 99) & (
                        df_['plan_repay_date'] >= (datetime.datetime.now() - datetime.timedelta(days=18)).strftime(
                            "%Y-%m-%d"))].reset_index(drop=True)
            _auc['date'] = pd.to_datetime(_auc['plan_repay_date']).apply(lambda x: x.strftime("%m-%d"))
            roc_auc_result = []
            for l in sorted(list(set(_auc['date']))):
                try:
                    roc_auc_result.append(
                        roc_auc_score(_auc[_auc['date'] == l]['is_overdue_'], _auc[_auc['date'] == l][i]))
                except:
                    roc_auc_result.append(0)
            line.add(j, sorted(list(set(_auc['date']))), roc_auc_result,
                     is_smooth=True, is_label_show=False,
                     line_opacity=0.2,
                     area_opacity=0.4,
                     is_random=True,
                     is_symbol_show=False)
        page.add(line)

    for l in ['card_op_only', 'card_woe_getui']:
        for m in ['final_model_pred', 'operator_model_pred']:
            overlap = Overlap(width=900)
            bar = Bar(l+' + '+m, "一审模型bin", width=900)

            fen = raw[raw['modeltype'] == l]
            fen = raw[['final_model_pred', 'operator_model_pred'] + ['is_overdue', 'plan_repay_date', 'status']]
            fen = fen[fen['plan_repay_date'].notnull()]
            fen = fen[fen['plan_repay_date'] < old_new_time].reset_index(drop=True)
            fen['is_overdue_'] = (fen['status'] != 100) * 1
            fen['fm'] = pd.qcut(fen[m], 10)
            fen_ = fen.groupby('fm')['is_overdue_'].agg(['mean', 'count'])
            fen_ = fen_[fen_['count'] > 20].reset_index()
            fen_.loc[:, 'fm'] = fen_.loc[:, 'fm'].apply(lambda x: "(%s, %s]" % (x.left, x.right)).values.tolist()

            bar.add('', list(fen_['fm']), list(fen_['count']), is_label_show=False, is_datazoom_show=True, is_random=True, datazoom_range=[0, 100], is_splitline_show=False)

            line = Line()
            line.add("逾期率", list(fen_['fm']), list(fen_['mean']), yaxis_min=0.2, line_width=1.5, is_smooth=True)

            overlap.add(bar)
            overlap.add(line, is_add_yaxis=True, yaxis_index=1)
            page.add(overlap)

    return render_template('pyecharts.html',
                           myechart=page.render_embed(),
                           # my_width="100%",
                           # my_height=600,
                           host='/static/js',
                           script_list=page.get_js_dependencies())


if __name__ == '__main__':
    raw = pd.read_csv("test.csv")
    raw['app_count'] = raw["app_pkg"].apply(lambda x: get_app_count(x))
    raw['gender'] = raw['gender'].apply(lambda x: get_gender(x))
    id_features = ['order_id', 'phone', 'inner_sources', 'create_time', 'plan_repay_date', 'is_overdue']
    xj_model_col = ['age', 'gender', 'op_lasts', 'gsm_active_night_hours_ratio_180_sum',
                    'history_meal_fee_mean_180_min', 'gsm_active_afternoon_hours_ratio_180_min',
                    'gsm_active_night_hours_ratio_180_std', 'gsm_active_evening_hours_ratio_180_sum',
                    'gsm_active_morning_hours_ratio_180_std', 'gsm_active_evening_hours_ratio_180_std',
                    'gsm_active_night_hours_ratio_180_max', 'sms_active_morning_hours_ratio_180_mean',
                    'gsm_communication_way_dail_180_std', 'sms_communication_way_received_180_skew',
                    'gsm_active_evening_hours_ratio_180_max', 'gsm_communication_way_dailed_180_skew',
                    'sms_active_night_hours_ratio_180_max', 'history_meal_fee_mean_180_mean',
                    'sms_active_evening_hours_ratio_180_max', 'gsm_active_afternoon_hours_180_mean',
                    'gsm_active_afternoon_hours_ratio_180_sum', 'sms_communication_way_received_180_std',
                    'gsm_active_times_180_std', 'gsm_active_afternoon_hours_180_skew', 'sms_active_times_180_skew',
                    'sms_active_morning_hours_ratio_180_sum', 'sms_communication_way_send_180_skew',
                    'gsm_active_night_hours_180_sum']
    ff_model_col = ['age', 'gender', 'op_lasts', 'account_balance', 'gsm_active_night_hours_ratio_180_sum',
                    'history_meal_fee_mean_180_min', 'gsm_active_afternoon_hours_ratio_180_min',
                    'gsm_active_night_hours_ratio_180_std', 'gsm_active_evening_hours_ratio_180_sum',
                    'gsm_active_morning_hours_ratio_180_std', 'gsm_active_evening_hours_ratio_180_std',
                    'gsm_active_night_hours_ratio_180_max', 'sms_active_morning_hours_ratio_180_mean',
                    'gsm_communication_way_dail_180_std', 'gsm_active_evening_hours_ratio_180_max',
                    'sms_active_night_hours_ratio_180_max', 'history_meal_fee_mean_180_mean',
                    'sms_active_evening_hours_ratio_180_max', 'gsm_active_afternoon_hours_180_mean',
                    'gsm_active_afternoon_hours_ratio_180_sum', 'sms_communication_way_received_180_std',
                    'gsm_active_times_180_std', 'sms_active_morning_hours_ratio_180_sum', 'history_other_fee_180_min',
                    'gsm_active_night_hours_180_sum']
    markcode_col = ['loan_app_nums', 'device_related_id_nums_30', 'device_related_mobile_nums_30',
                    'mobile_related_id_nums_30', 'mobile_related_device_nums_30', 'mobile_qz_current_overdue',
                    'mobile_qz_amount_not_clear', 'mobile_qz_history_overdue_nums', 'mobile_superloan_current_overdue',
                    'mobile_superloan_amount_not_clear', 'mobile_superloan_history_overdue_nums',
                    'mobile_bestbuy_current_overdue', 'mobile_bestbuy_amount_not_clear',
                    'mobile_bestbuy_history_overdue_nums', 'mobile_car_finance_current_overdue',
                    'mobile_car_finance_amount_not_clear', 'mobile_car_finance_history_overdue_nums',
                    'mobile_fan_card_current_overdue', 'mobile_fan_card_amount_not_clear',
                    'mobile_fan_card_history_overdue_nums', 'mobile_finup_lend_current_overdue',
                    'mobile_finup_lend_amount_not_clear', 'mobile_finup_lend_history_overdue_nums',
                    'id_related_device_nums_30', 'id_related_mobile_nums_30', 'id_qz_current_overdue',
                    'id_qz_amount_not_clear', 'id_qz_history_overdue_nums', 'id_superloan_amount_not_clear',
                    'id_superloan_history_overdue_nums', 'id_bestbuy_current_overdue', 'id_bestbuy_amount_not_clear',
                    'id_bestbuy_history_overdue_nums', 'id_car_finance_current_overdue',
                    'id_car_finance_amount_not_clear', 'id_car_finance_history_overdue_nums',
                    'id_fan_card_current_overdue', 'id_fan_card_amount_not_clear', 'id_fan_card_history_overdue_nums',
                    'id_finup_lend_current_overdue', 'id_finup_lend_amount_not_clear',
                    'id_finup_lend_history_overdue_nums', 'phonebook_contacts_nums', 'phonebook_finup_registered_ratio',
                    'phonebook_finup_history_overdue_nums', 'phonebook_finup_current_overdue_nums',
                    'emergency_finup_current_overdue_nums', 'emergency_finup_history_overdue_nums',
                    'operator_account_days', 'dailed_record_nums_30']
    sanfang = ['black_overduegrade', 'um_credit_score', 'app_stability_7d', 'loan_7d', 'finance_7d', 'tcrisk_score',
               'rule_final_weight', 'apply_latest_one_month', 'behavior_loans_score']
    important_features = ['age', 'gender', 'app_count', 'op_lasts', 'gsm_active_night_hours_ratio_180_sum', 'black_overduegrade',
                          'um_credit_score', 'loan_7d', 'apply_latest_one_month',
                          'operator_dailed_contacts_top20_30_finup_registered_nums', 'short_dail_ratio_30',
                          'behavior_loans_settle_count', 'behavior_history_fail_fee',
                          'current_consfin_org_count', 'behavior_latest_one_month_suc', 'rule_final_weight',
                          'tcrisk_score', 'current_consfin_max_limit', 'apply_latest_six_month',
                          'behavior_loans_overdue_count', 'behavior_history_suc_fee',
                          'phonebook_contacts_nums', 'history_month_cnt_180_sum', 'gsm_communication_way_dail_180_sum', 'gsm_communication_way_dailed_180_sum']
    old_period = '2019-02-20,2019-04-01'
    new_period = '2019-04-01,2019-04-11'
    old_plan_repay_date = '2019-02-01'
    new_plan_repay_date = '2019-03-20'
    old_steps = -1
    new_steps = -1
    td = 5
    abnormal_threshold = 0.2

    import_col = ['create_time'] + important_features + ['ipprovince', 'gps_province']
    import_old_time = old_period.split(",")
    import_current_time = new_period.split(",")
    import_old_time_space = str(old_period)
    import_current_time_space = str(new_period)
    import_data_raw = raw[import_col]

    start_point = raw[raw['create_time'] >= old_period.split(',')[0]]
    start_point['user_tag'] = start_point['user_source'].apply(lambda x: 'new' if x == 'new_user' else 'old')
    start_point = start_point[~(start_point['tcrisk_score'].str.contains(',') == True)]
    start_point['tcrisk_score'] = start_point['tcrisk_score'].astype(float)

    import_data = pd.DataFrame()
    import_data = pd.concat([import_data, features_split_time(import_data_raw, import_old_time, '-1', 'old')])
    import_data = pd.concat([import_data, features_split_time(import_data_raw, import_current_time, '-1', 'current')])

    import_data['ipprovince'] = import_data['ipprovince'].fillna('未知')
    import_data['gps_province'] = import_data['gps_province'].fillna('未知')

    old_import_data = import_data[import_data['time_flag'] == 'old'].reset_index(drop=1)
    current_import_data = import_data[import_data['time_flag'] == 'current'].reset_index(drop=1)

    old_import_data = old_import_data[~(old_import_data['tcrisk_score'].str.contains(',') == True)]
    current_import_data = current_import_data[~(current_import_data['tcrisk_score'].str.contains(',') == True)]
    old_import_data['tcrisk_score'] = old_import_data['tcrisk_score'].astype(float)
    current_import_data['tcrisk_score'] = current_import_data['tcrisk_score'].astype(float)

    tbp = TableBinPsi()
    table_bins_cross = tbp.fit(old_import_data, current_import_data,
                               list(set(important_features) - {'age', 'gender'}), 15, 10, 10)
    psi_result = tbp.transform(list(set(important_features) - {'age', 'gender'}))
    abnormal_list = list(psi_result[psi_result['psi'] > abnormal_threshold]['varnames'])

    tt = Monitor(list(set(['age', 'gender', 'app_count', 'phonebook_contacts_nums', 'history_month_cnt_180_sum', 'gsm_communication_way_dail_180_sum', 'gsm_communication_way_dailed_180_sum'] + abnormal_list)), old_period, new_period, old_time_space=old_steps,
                 current_time_space=new_steps)
    tt.data_analysis(raw)
    tmp = tt.rel_df_show

    old_new_time = datetime.datetime.now().strftime("%Y-%m-%d")
    df_ = raw[raw['plan_repay_date'].notnull()]
    df_ = df_[df_['plan_repay_date'] < old_new_time].reset_index(drop=True)
    df_ = df_[df_['create_time'] >= '2018-12-22'].reset_index(drop=True)
    df_['is_overdue_'] = (df_['status'] != 100) * 1
    df_['user_tag'] = df_['user_source'].apply(lambda x: 'new' if x == 'new_user' else 'old')
    old = df_[(df_['plan_repay_date'] >= old_plan_repay_date) & (df_['plan_repay_date'] < new_plan_repay_date)].reset_index(drop=True)
    new = df_[
        (df_['plan_repay_date'] >= (datetime.datetime.now() - datetime.timedelta(days=td)).strftime("%Y-%m-%d")) & (
                    df_['plan_repay_date'] < old_new_time)].reset_index(drop=True)

    raw_ = raw[raw['plan_repay_date'].notnull()]
    raw_['is_overdue'] = (raw_['status'] != 100) * 1
    raw_['is_overdue'] = 1 - raw_['is_overdue']
    oo = raw_[(raw_['plan_repay_date'] >= (datetime.datetime.now() - datetime.timedelta(days=18)).strftime("%Y-%m-%d")) & (raw_['plan_repay_date'] < old_new_time)][
        ['plan_repay_date', 'operator_model_pred', 'is_overdue']]
    oo = oo[oo['operator_model_pred'] < 99].reset_index(drop=True)
    oo['date'] = pd.to_datetime(oo['plan_repay_date']).apply(lambda x: x.strftime("%m-%d"))

    app.run(
        host='0.0.0.0',
        port=8080,
        debug=True
            )
