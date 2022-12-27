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
    # 将数据放到合成数据集目录中，用于训练
    for mode in ["train", "val"]:
        to_sub_dir = os.path.join(dst_dir, mode)
        os.makedirs(to_sub_dir, exist_ok=False)
        # if os.path.exists(to_sub_dir):
        #     shutil.rmtree(to_sub_dir)
        # os.mkdir(to_sub_dir)

        os.system(f"cp -r {data_dir}/{mode}/images_patched {to_sub_dir}")
        os.system(f"cp -r {data_dir}/{mode}/annotations_patched {to_sub_dir}/")

if __name__ == "__main__":
    data_dir = '/home/lucien/research/lucienresearch/OriRCNNDet/mmrotate/data/datasets_hw'
    dst_dir = "/home/lucien/research/lucienresearch/OriRCNNDet/mmrotate/data/datasets_synthesis"
    # copy_patch_data(data_dir, dst)

    format2dota(dst_dir)