import os
import shutil
def format2dota(data_dir):
    """
    将 xray 的 annotations 中标注文件的格式转换为 DOTA 格式

    train00001.jpg 1 battery 377 19 546 139 390 6 553 41 532 144 368 109
    """

    for mode in ["train", "val"]:

        ann_dir = os.path.join(data_dir, f'{mode}/annotations/')
        ann_dota_dir = os.path.join(data_dir, f'{mode}/annotations_dota/')

        # 获取标注文件列表
        ann_files = [ann_file for ann_file in os.listdir(ann_dir) if ann_file.endswith('.txt')]
        os.makedirs(ann_dota_dir, exist_ok=True)

        for ann_file in ann_files:
            # 读取数据，修改为 SOTA 格式
            with open(os.path.join(ann_dir, ann_file), 'r') as ff:
                with open(os.path.join(ann_dota_dir, ann_file), 'w') as fw:
                    for line in ff.readlines():
                        try:
                            arr = line.strip('\n').split(' ')
                            category = arr[2]
                            points = " ".join(arr[-8:])
                            difficult = 0
                            fw.write("{} {} {}\n".format(points, category, difficult))
                        except:
                            print("occur error:", line)


def copy_patch_data(data_dir, dst_dir):
    """ 将数据放到合成数据集目录中，用于训练 """

    # for mode in ["train", "val", "test"]:
    for mode in ["val"]:
        sub_dir = os.path.join(dst_dir, mode)
        os.makedirs(os.path.join(sub_dir, 'images'), exist_ok=False)
        os.makedirs(os.path.join(sub_dir, 'annotations'), exist_ok=False)

        if mode == 'train':
            os.system(f"cp -r {data_dir}/{mode}/images_patched/* {sub_dir}/images/")
            os.system(f"cp -r {data_dir}/{mode}/annotations_patched/* {sub_dir}/annotations/")
        else:
            os.system(f"cp -r {data_dir}/{mode}/images {sub_dir}/")
            os.system(f"cp -r {data_dir}/{mode}/annotations {sub_dir}/")


def merge_results(res_dir):
    """ 对多个输出结果进行合并 """
    import numpy as np
    import os

    classes = ['knife', 'pressure', 'umbrella', 'lighter', 'OCbottle', 'glassbottle', 'battery', 'metalbottle', 'electronicequipment']
    data = [name for name in os.listdir(res_dir) if name.startswith('Task1_')]

    print(data)

    result_all = []
    for file in data:
        name = file.split('.')[0].split('_')[1]
        assert name in classes, 'category not in classes'
        with open(os.path.join(res_dir, file), 'r', encoding='utf-8') as f:
            txt = f.readlines()
            for t in txt:
                result = []
                val = t.rstrip().split(" ")
                result.append(val[0] + '.jpg')
                result.append(name)
                result.extend(val[1:])
                result_all.append(result)

    data_result = np.array(result_all)
    np.savetxt(os.path.join(res_dir,'result.txt'), data_result[np.lexsort(data_result[:, ::-1].T)], '%s')




if __name__ == "__main__":
    root_dir = '/home/lucien/research/lucienresearch/OriRCNNDet/mmrotate/'
    # 原始数据集目录
    data_dir = root_dir + 'data/datasets_hw'
    # format2dota(data_dir)
    
    # 将 datasets_hw 中合成的数据复制到 datasets_synthesis 中
    dst_dir = root_dir + "data/datasets_synthesis"
    # copy_patch_data(data_dir, dst_dir)  # copy 数据
    format2dota(dst_dir)  # 转换数据格式到 dota
    
    # 对预测结果进行合并（预测结果是按每个类别一个文件进行保存的）
    res_dir = root_dir + 'work_dirs/oriented_rcnn_swin_tiny_fpn_1x_dota_le90_xray/result/'
    # merge_results(res_dir)

    