# OriRCNNDet

### Update:
2022-12-8 引入 mmrotate repository


### 将 xray 的 annotations 中标注文件的格式转换为 DOTA 格式  
把 datasets_hw.zip 解压到 OriRCNNDet/mmrotate/data/ 中
执行 mmrotate/data/data_prep.py 转换格式


### 为 xray 数据添加配置文件  
mmrotate/configs/_base_/datasets/xray.py  
mmrotate/configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90_xray.py 


### 执行训练脚本
python tools/train.py  configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90_xray.py


### 验证集评估结果
2022-12-8 在 val set 上评估
| class       | gts | dets | recall | ap    |
|-------------|-----|------|--------|-------|
| knife       | 693 | 1901 | 0.667  | 0.540 |
| pressure    | 173 | 784  | 0.861  | 0.776 |
| umbrella    | 110 | 469  | 0.845  | 0.787 |
| lighter     | 461 | 1422 | 0.588  | 0.487 |
| OCbottle    | 281 | 877  | 0.868  | 0.768 |
| glassbottle | 133 | 667  | 0.812  | 0.716 |
| battery     | 303 | 1069 | 0.842  | 0.705 |
| metalbottle | 250 | 854  | 0.904  | 0.820 |
| mAP         |     |      |        | 0.700 |