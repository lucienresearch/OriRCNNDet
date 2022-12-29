# OriRCNNDet

### Update:
2022-12-8 引入 mmrotate repository


### 运行环境
pip install openmim  
mim install mmcv-full  
mim install mmdet  
cd mmrotate  
pip install -r requirements/build.txt  
pip install -v -e .  


### 将 xray 的 annotations 中标注文件的格式转换为 DOTA 格式  
把 datasets_hw.zip 解压到 OriRCNNDet/mmrotate/data/ 中  
执行 mmrotate/data/data_prep.py 转换格式  


### 为 xray 数据添加配置文件  
mmrotate/configs/_base_/datasets/xray.py   
NOTE: 在 xray.py 中，将 data_root 修改为具体的数据集目录  
mmrotate/configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90_xray.py   


### 执行训练脚本
python tools/train.py configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90_xray.py    
python tools/train.py configs/oriented_rcnn/oriented_rcnn_swin_tiny_fpn_1x_dota_le90_xray.py (更优)

### 执行测试脚本
python tools/test.py configs/oriented_rcnn/oriented_rcnn_swin_tiny_fpn_1x_dota_le90_xray.py work_dirs/oriented_rcnn_swin_tiny_fpn_1x_dota_le90_xray/epoch_30.pth  
--format-only --eval-options submission_dir=work_dirs/oriented_rcnn_swin_tiny_fpn_1x_dota_le90_xray/result  


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

2022-12-27 23:06:06,077 - val set - 扩充数据之后
可以看出，由于增加了 gt 的数量，精度降低了。
| class               | gts | dets | recall | ap    |
|---------------------|-----|------|--------|-------|
| knife               | 693 | 2046 | 0.645  | 0.497 |
| pressure            | 236 | 1257 | 0.818  | 0.597 |
| umbrella            | 173 | 844  | 0.763  | 0.599 |
| lighter             | 526 | 2174 | 0.648  | 0.481 |
| OCbottle            | 350 | 1366 | 0.871  | 0.665 |
| glassbottle         | 203 | 923  | 0.862  | 0.578 |
| battery             | 368 | 1718 | 0.848  | 0.606 |
| metalbottle         | 302 | 1234 | 0.930  | 0.738 |
| electronicequipment | 444 | 1152 | 0.793  | 0.690 |
| mAP                 |     |      |        | 0.606 |


2022-12-29 05:02:15,345 - mmrotate - INFO - 
epoch 30，在 epoch 12 时已经基本收敛完成
这里没有对 val set 进行增强
2022-12-29 05:02:15,357 - mmrotate - INFO - Exp name: oriented_rcnn_swin_tiny_fpn_1x_dota_le90_xray.py
test set mAP 0.763
| class               | gts | dets | recall | ap    |
|---------------------|-----|------|--------|-------|
| knife               | 693 | 1515 | 0.749  | 0.650 |
| pressure            | 173 | 513  | 0.884  | 0.778 |
| umbrella            | 110 | 197  | 0.909  | 0.877 |
| lighter             | 461 | 982  | 0.651  | 0.560 |
| OCbottle            | 281 | 551  | 0.911  | 0.852 |
| glassbottle         | 133 | 306  | 0.857  | 0.775 |
| battery             | 303 | 738  | 0.855  | 0.733 |
| metalbottle         | 250 | 559  | 0.916  | 0.848 |
| electronicequipment | 444 | 747  | 0.845  | 0.789 |
| mAP                 |     |      |        | 0.763 |
