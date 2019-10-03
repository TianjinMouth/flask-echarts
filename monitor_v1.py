# 处理特征时间分段的函数
"""
df:需要分段的原始数据
time:起始时间
time_space:时间间隔
flag:当前数据的标志，后边分bin用
bin_col:需要分段的列
"""

# 检查数据类型
def check_dtype(tdf, varlist, var_diff_num):
    df = tdf.copy()
    cate_f = []
    con_f = []
    con_copy = []
    for a in varlist:
        try:
            df[a] = df[a].astype('float')
            con_f.append(a)
        except ValueError:
            cate_f.append(a)
    for a in con_f:
        if df[a].nunique() <= var_diff_num:
            cate_f.append(a)
        else:
            con_copy.append(a)
    con_f = con_copy
    return df,cate_f,con_f


#计算某个特征的mean,std的
def get_mean_std(tdf, col, bin_col):
    df = tdf.copy()
    temp_mean = tdf.groupby([bin_col]).apply(lambda x: x[col].mean()).reset_index()
    temp_mean.columns = [bin_col, 'gg']
    temp_mean.insert(1, col, col+'=mean')
    temp_mean.insert(1, 'f_type', 'desc')
    temp_mean.insert(1, 'var_type', 'conti')
    temp_std = tdf.groupby([bin_col]).apply(lambda x: x[col].std()).reset_index()
    temp_std.columns = [bin_col, 'gg']
    temp_std.insert(1, col, col+'=std')
    temp_std.insert(1, 'f_type', 'desc')
    temp_std.insert(1, 'var_type', 'conti')
    return pd.concat([temp_mean, temp_std])


class Monitor(object):
    def __init__(self, cols, old_time, current_time,
                 channel_name=['xianjincard'],
                 old_time_space=-1,
                 current_time_space=-1,
                 bin_num=5,
                 var_diff_num=10,
                 cut_col='create_time'):

        self.cols = ['create_time'] + cols
        self.bin_num = bin_num  # 连续变量分箱数
        self.var_diff_num = var_diff_num    #
        self.cut_col = cut_col
        self.old_time = old_time.split(",")
        self.current_time = current_time.split(",")
        self.old_time_space = str(old_time_space)
        self.current_time_space = str(current_time_space)
        self.outlier_bound = None
        self.channel_name = channel_name
        self.rel_df = None
        self.rel_df_show = None
        self.sample_size = None
        self.channel_size = {}
        self.categorical_cols = None
        self.continuous_cols = None

    '''重要特征的分析'''
    def data_analysis(self, df):
        # 1、加载重要特征需要分析的列，若是没有配置，则默认为用户的基本信息为重要特征
        self.sample_size = df.shape[0]

        if len(self.cols) == 0:
            print("您没有定义需要分析的重要特征,自动默认用户的基本信息为重要特征")
            data_raw = df[['age', 'gender']]
        else:
            data_raw = df[self.cols]

        if self.cut_col == 'inner_sources':
            data = pd.DataFrame()
            start_time = self.old_time[0]
            end_time = self.current_time[1]
            for i in range(len(self.channel_name)):
                temp_data = data_raw[(data_raw[self.cut_col] == self.channel_name[i])
                                     & (data_raw['create_time'] >= start_time)
                                     & (data_raw['create_time'] < end_time)]
                data = pd.concat([data, temp_data])
        else:
            data = pd.DataFrame()
            data = pd.concat([data, self.features_split_time(data_raw, self.old_time, self.old_time_space, 'old')])
            data = pd.concat([data, self.features_split_time(data_raw, self.current_time, self.current_time_space, 'current')])
            self.cut_col = 'time_flag'
            self.channel_name = ['old', 'current']

        for c in self.channel_name:
            self.channel_size[c] = data[data[self.cut_col] == c].shape[0]
        self.rel_df = self.analysis_import_channel_features(data)

        # 异常值
        self.data = data
        outlier = self.outlier(data)
        self.rel_df = pd.concat([self.rel_df, outlier])
        self.rel_df = self.rel_df.sort_values(["var", "cut_col", "var_type"])
        self.rel_df_show = pd.pivot_table(self.rel_df.reset_index(), index=['bins', 'var'], columns=['cut_col']
                                          , values=["value"]).reset_index()
        self.rel_df_show = self.rel_df_show.sort_values(['var', 'bins'])

    def features_split_time(self, tdf, time, time_space, flag):

        df = tdf.copy()
        temp_import_data = pd.DataFrame()
        if time_space == '-1':
            import_data = df[(df[self.cut_col] >= time[0]) & (df[self.cut_col] < time[1])]
            import_data['time_flag'] = flag
            temp_import_data = pd.concat([temp_import_data, import_data])
        else:
            import_time_list = [str(t)[:10] for t in
                                pd.date_range(start=time[0], end=time[1], freq=time_space + 'D', name='dt')]
            if time[1] not in import_time_list:
                import_time_list.append(time[1])
            for i in range(len(import_time_list) - 1):
                import_data = df[(df[self.cut_col] >= import_time_list[i]) & (df[self.cut_col] < import_time_list[i + 1])]
                # import_data['time_bz'] = import_time_list[i] + "_" + import_time_list[i + 1]
                import_data['time_flag'] = flag
                temp_import_data = pd.concat([temp_import_data, import_data])
        return temp_import_data

    def analysis_import_channel_features(self, df):
        all_col = df.columns.tolist()
        temp_set = set()
        temp_set.add(self.cut_col)
        temp_set.add('create_time')

        tdf, cate_f, con_f = check_dtype(df, list(set(all_col) - temp_set), self.var_diff_num)
        self.categorical_cols = cate_f
        self.continuous_cols = con_f
        cate_f_df = pd.DataFrame()
        conti_f_df = pd.DataFrame()
        # 处理离散变量
        for a in cate_f:
            df_nan = self.get_nan(tdf, a, 'cate')
            notnull_df = tdf[tdf[a].notnull()]
            rename_less_cat = pd.DataFrame(pd.value_counts(notnull_df[a]))
            if rename_less_cat.shape[0] > self.bin_num:
                for rnum in range((self.bin_num - 1), len(rename_less_cat)):
                    rena = rename_less_cat.index[rnum]
                    notnull_df[a] = notnull_df[a].replace(rena, 'others')

            gg = notnull_df.groupby([self.cut_col, a], as_index=0)['create_time'].count()
            gg.columns = ['cut_col', 'bins', 'value']
            for c in self.channel_name:
                gg.loc[gg['cut_col'] == c, 'value'] = gg.loc[gg['cut_col'] == c, 'value'] \
                                                      / self.channel_size[c]
            gg.insert(0, 'var', a)
            gg.insert(1, 'var_type', 'cate')
            temp_merge = pd.concat([gg, df_nan], axis=0, ignore_index=True)
            cate_f_df = pd.concat([cate_f_df, temp_merge], ignore_index=True)

        # 处理连续变量
        for a in con_f:
            df_nan = self.get_nan(tdf, a, 'conti')
            temp_tdf = tdf[tdf[a].notnull()]
            gg = self.get_cut_df(temp_tdf, a)
            temp_merge = pd.concat([gg, df_nan], axis=0, ignore_index=True)
            conti_f_df = pd.concat([conti_f_df, temp_merge])

        import_rel = pd.concat([cate_f_df, conti_f_df])
        import_rel['value'] = import_rel['value'].fillna(0.0)
        return import_rel

    def outlier(self, df):
        # 异常值
        df_con = df[[self.cut_col] + self.continuous_cols]
        df_cat = df[[self.cut_col] + self.categorical_cols]
        # 连续变量
        old_data = df_con[df_con[self.cut_col] == self.channel_name[0]].reset_index(drop=1)
        old_data = old_data.drop(self.cut_col, axis=1)
        old_outlier = self.cal_outlier_nums(old_data, cols=old_data.columns.tolist())
        old_outlier.insert(0, 'cut_col', self.channel_name[0])

        rdf = pd.DataFrame()
        rdf = pd.concat([rdf, old_outlier])
        for i in range(1, len(self.channel_name)):
            new_data = df_con[df_con[self.cut_col] == self.channel_name[i]].reset_index(drop=1)
            new_data = new_data.drop(self.cut_col, axis=1)
            new_outlier = self.cal_outlier_nums(new_data, upper=old_outlier['upper_bound'],
                                                lower=old_outlier['lower_bound'], cols=old_data.columns.tolist())
            new_outlier.insert(0, 'cut_col', self.channel_name[i])
            rdf = pd.concat([rdf, new_outlier])

        rdf = rdf.reset_index().rename(columns={'index': 'var'})
        rdf.insert(2, 'var_type', 'conti')

        # 分类变量
        old_data = df_cat[df_cat[self.cut_col] == self.channel_name[0]].reset_index(drop=1)
        old_data = old_data.drop(self.cut_col, axis=1)
        unique_values = {}
        cat_outlier = []
        for c in self.categorical_cols:
            unique_values[c] = old_data[c].unique().tolist()
            cat_outlier.append([c, self.channel_name[0], 'cate', 0, 0, 0, 0])

        for i in range(1, len(self.channel_name)):
            new_data = df_cat[df_cat[self.cut_col] == self.channel_name[i]].reset_index(drop=1)
            for c in self.categorical_cols:
                _diff_values = [v for v in new_data[c].unique().tolist() if v not in unique_values[c]]
                iso_nums = new_data[new_data[c].isin(_diff_values)][c].count()
                value = iso_nums / self.channel_size[self.channel_name[i]]
                cat_outlier.append([c, self.channel_name[i], 'cate', 0, 0, iso_nums, value])

        cat_outlier = pd.DataFrame(cat_outlier, columns=rdf.columns)

        rdf = pd.concat([rdf, cat_outlier])
        rdf.insert(4, 'bins', 'outlier')
        rdf = rdf[['var', 'var_type', 'cut_col', 'bins', 'value']]
        return rdf

    # 计算某个特征的空值的
    def get_nan(self, tdf, col, col_type):
        df = tdf.copy()
        temp_nan = df.groupby(self.cut_col).apply(lambda x: x[col].isnull().sum()).reset_index()
        temp_nan.columns = ['cut_col', 'value']
        # 除各自分母
        for c in self.channel_name:
            temp_nan.loc[temp_nan['cut_col'] == c, 'value'] = temp_nan.loc[temp_nan['cut_col'] == c, 'value'] \
                                                              / self.channel_size[c]
        temp_nan.insert(0, 'var', col)
        temp_nan.insert(1, 'var_type', col_type)
        temp_nan.insert(3, 'bins', 'nan')
        return temp_nan

    # 对某分数据cut进行计算
    def get_cut_df(self, tdf, col):
        df = tdf.copy()
        df_old = df[df[self.cut_col] == self.channel_name[0]]
        df_old[col], ind_index = pd.cut(df_old[col], self.bin_num, retbins=True)
        temp_df = pd.DataFrame()
        temp_df = pd.concat([temp_df, df_old])
        for i in range(1, len(self.channel_name)):
            df_new = df[df[self.cut_col] == self.channel_name[i]]
            df_new[col] = pd.cut(df_new[col], bins=ind_index)
            temp_df = pd.concat([temp_df, df_new])

        gg = temp_df.groupby([self.cut_col, col], as_index=0)['create_time'].count()
        gg.columns = ['cut_col', 'bins', 'value']
        gg['bins'] = gg['bins'].apply(lambda x: "(%s,%s]" % (x.left, x.right))
        for c in self.channel_name:
            gg.loc[gg['cut_col'] == c, 'value'] = gg.loc[gg['cut_col'] == c, 'value'] \
                                                              / self.channel_size[c]
        gg.insert(0, 'var', col)
        gg.insert(1, 'var_type', 'cate')
        return gg

    @staticmethod
    def cal_outlier_nums(df, cols, upper=None, lower=None):
        # 连续变量求异常值
        tmp_df = df[cols].copy()
        mean_values = tmp_df.mean()
        std_values = tmp_df.std()
        if upper is None:
            upper_bound = mean_values + 3 * std_values
            lower_bound = mean_values - 3 * std_values
        else:
            upper_bound = upper
            lower_bound = lower

        res = pd.concat([upper_bound, lower_bound], axis=1)
        res.columns = ['upper_bound', 'lower_bound']
        nums = df.shape[0]

        iso_nums = []
        iso_ratio = []
        for col in tmp_df.columns:
            _tmp = tmp_df[(tmp_df[col] > upper_bound[col]) | (tmp_df[col] < lower_bound[col])].shape[0]
            iso_nums.append(_tmp)
            iso_ratio.append(_tmp / nums)
        res['iso_nums'] = iso_nums
        res['value'] = iso_ratio
        return res


def feature_bar_plot(df, name='age'):
    old_legend = df['cut_col'].unique()[0]
    new_legend = df['cut_col'].unique()[1]
    old = df[(df['cut_col'] == old_legend) & (df['var'] == name)][['bins', 'value']]
    new = df[(df['cut_col'] == new_legend) & (df['var'] == name)][['bins', 'value']]
    tmp = old.merge(new, on='bins', how='left')
    old_cuts = tmp['value_x']
    new_cuts = tmp['value_y'].fillna(0.0)

    cat_name = df[(df['time'] == old_legend) & (df['var'] == name)]['bins']
    show_diff_categorical(old_cuts, new_cuts, cat_name, name, rotation=0)


def show_diff_categorical(y1, y2, cols, title, rotation=0):
    plt.figure(figsize=(10, 5))
    total_width, n = 0.8, 2.5
    width = total_width / n
    x1 = []
    for k,v in enumerate(cols):
        x1.append(k)
    plt.bar(x1, y1, width=width, label='old', fc='#ED9121')
    for i in range(len(x1)):
        x1[i] = x1[i] + width
    plt.bar(x1, y2, width=width, label='new', fc='#3D59AB')
    plt.xlabel('categories')
    plt.ylabel('rate')
    plt.title(title)
    plt.xticks(np.arange(len(cols)), cols, rotation=rotation)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    raw = pd.read_csv("/Users/finup/Project/MoonLight/data/v20190311/t.csv")

    id_features = ['order_id', 'phone', 'inner_sources', 'create_time', 'plan_repay_date', 'is_overdue']
    xj_model_col = ['age', 'gender', 'op_lasts', 'gsm_active_night_hours_ratio_180_sum', 'history_meal_fee_mean_180_min', 'gsm_active_afternoon_hours_ratio_180_min', 'gsm_active_night_hours_ratio_180_std', 'gsm_active_evening_hours_ratio_180_sum', 'gsm_active_morning_hours_ratio_180_std', 'gsm_active_evening_hours_ratio_180_std', 'gsm_active_night_hours_ratio_180_max', 'sms_active_morning_hours_ratio_180_mean', 'gsm_communication_way_dail_180_std', 'sms_communication_way_received_180_skew', 'gsm_active_evening_hours_ratio_180_max', 'gsm_communication_way_dailed_180_skew', 'sms_active_night_hours_ratio_180_max', 'history_meal_fee_mean_180_mean', 'sms_active_evening_hours_ratio_180_max', 'gsm_active_afternoon_hours_180_mean', 'gsm_active_afternoon_hours_ratio_180_sum', 'sms_communication_way_received_180_std', 'gsm_active_times_180_std', 'gsm_active_afternoon_hours_180_skew', 'sms_active_times_180_skew', 'sms_active_morning_hours_ratio_180_sum', 'sms_communication_way_send_180_skew', 'gsm_active_night_hours_180_sum']
    ff_model_col = ['age', 'gender', 'op_lasts', 'account_balance', 'gsm_active_night_hours_ratio_180_sum', 'history_meal_fee_mean_180_min', 'gsm_active_afternoon_hours_ratio_180_min', 'gsm_active_night_hours_ratio_180_std', 'gsm_active_evening_hours_ratio_180_sum', 'gsm_active_morning_hours_ratio_180_std', 'gsm_active_evening_hours_ratio_180_std', 'gsm_active_night_hours_ratio_180_max', 'sms_active_morning_hours_ratio_180_mean', 'gsm_communication_way_dail_180_std', 'gsm_active_evening_hours_ratio_180_max', 'sms_active_night_hours_ratio_180_max', 'history_meal_fee_mean_180_mean', 'sms_active_evening_hours_ratio_180_max', 'gsm_active_afternoon_hours_180_mean', 'gsm_active_afternoon_hours_ratio_180_sum', 'sms_communication_way_received_180_std', 'gsm_active_times_180_std', 'sms_active_morning_hours_ratio_180_sum', 'history_other_fee_180_min', 'gsm_active_night_hours_180_sum']
    markcode_col = ['loan_app_nums', 'device_related_id_nums_30', 'device_related_mobile_nums_30', 'mobile_related_id_nums_30', 'mobile_related_device_nums_30', 'mobile_qz_current_overdue', 'mobile_qz_amount_not_clear', 'mobile_qz_history_overdue_nums', 'mobile_superloan_current_overdue', 'mobile_superloan_amount_not_clear', 'mobile_superloan_history_overdue_nums', 'mobile_bestbuy_current_overdue', 'mobile_bestbuy_amount_not_clear', 'mobile_bestbuy_history_overdue_nums', 'mobile_car_finance_current_overdue', 'mobile_car_finance_amount_not_clear', 'mobile_car_finance_history_overdue_nums', 'mobile_fan_card_current_overdue', 'mobile_fan_card_amount_not_clear', 'mobile_fan_card_history_overdue_nums', 'mobile_finup_lend_current_overdue', 'mobile_finup_lend_amount_not_clear', 'mobile_finup_lend_history_overdue_nums', 'id_related_device_nums_30', 'id_related_mobile_nums_30', 'id_qz_current_overdue', 'id_qz_amount_not_clear', 'id_qz_history_overdue_nums', 'id_superloan_amount_not_clear', 'id_superloan_history_overdue_nums', 'id_bestbuy_current_overdue', 'id_bestbuy_amount_not_clear', 'id_bestbuy_history_overdue_nums', 'id_car_finance_current_overdue', 'id_car_finance_amount_not_clear', 'id_car_finance_history_overdue_nums', 'id_fan_card_current_overdue', 'id_fan_card_amount_not_clear', 'id_fan_card_history_overdue_nums', 'id_finup_lend_current_overdue', 'id_finup_lend_amount_not_clear', 'id_finup_lend_history_overdue_nums', 'phonebook_contacts_nums', 'phonebook_finup_registered_ratio', 'phonebook_finup_history_overdue_nums', 'phonebook_finup_current_overdue_nums', 'emergency_finup_current_overdue_nums', 'emergency_finup_history_overdue_nums', 'operator_account_days', 'dailed_record_nums_30']
    sanfang = ['black_overduegrade', 'um_credit_score', 'app_stability_7d', 'loan_7d', 'finance_7d', 'tcrisk_score', 'rule_final_weight', 'apply_latest_one_month', 'behavior_loans_score']
    important_features = ['age', 'gender', 'op_lasts', 'gsm_active_night_hours_ratio_180_sum', 'black_overduegrade', 'um_credit_score', 'loan_7d', 'apply_latest_one_month', 'operator_dailed_contacts_top20_30_finup_registered_nums', 'short_dail_ratio_30']
    third_model = ['behavior_loans_settle_count', 'behavior_history_fail_fee', 'current_consfin_org_count', 'tc_riskcv',
                   'behavior_latest_one_month_suc', 'um_credit_score', 'rule_final_weight', 'tcrisk_score',
                   'current_consfin_max_limit', 'device_brand', 'apply_latest_one_month', 'apply_latest_six_month',
                   'behavior_loans_overdue_count', 'behavior_history_suc_fee']

    old_period = '2019-03-01,2019-03-07'
    new_period = '2019-03-08,2019-03-13'
    old_steps = -1
    new_steps = -1
    abnormal_threshold = 0.05

    t = list(set(ff_model_col + third_model))
    im = Monitor(ff_model_col, old_period, new_period, old_time_space=old_steps, current_time_space=new_steps)
    im.data_analysis(raw)
    print(im.rel_df)
