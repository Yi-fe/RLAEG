import csv

original_v = []
a_list = []
modify_v = []
ac_list = []
mo_v = []

def write_results(ori_v, a_l, mod_v, ac_l):
    original_v.append(ori_v)
    a_list.append(a_l)
    modify_v.append(mod_v)
    ac_list.append(ac_l)
    dic = {}
    for i in range(len(ori_v)):
        dic[i] = abs(ori_v[i] - mod_v[i])
    mo_list = sorted(dic.items(), key=lambda x:x[1],reverse=True)
    mo_v.append(mo_list[:20])


def save_result():
    header = ['original_vector', 'action_xl', 'modified_vector', 'modify-origin', 'action_list']
    data = []
    for i in range(len(mo_v)):
        li = []
        li.append(original_v[i])
        li.append(ac_list[i])
        li.append(modify_v[i])
        li.append(mo_v[i])
        li.append(a_list[i])
        data.append(li)

    with open('result.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        # write the data
        writer.writerows(data)
