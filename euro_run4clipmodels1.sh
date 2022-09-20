# # 'RN50x4', input size 288*288
# # 'RN50x16' input size 384*384
# # 'RN50x64' 448*448
# 
root="/media/admin1/data1/image_scene_classification/"

python eurosat_fs.py \
    --shot 10 \
    --dataset 'EuroSATallBands' \
    --model_name "ViT-L/14" \
    --img_size 224 \
    --root $root
    
# python eurosat_fs.py \
#     --shot 10 \
#     --dataset 'EuroSATallBands' \
#     --model_name "RN50" \
#     --root $root

# python eurosat_fs.py \
#     --shot 10 \
#     --dataset 'EuroSATallBands' \
#     --model_name "RN101" \
#     --root $root

# python eurosat_fs.py \
#     --shot 10 \
#     --dataset 'EuroSATallBands' \
#     --model_name "RN50x4" \
#     --img_size 288 \
#     --root $root

# python eurosat_fs.py \
#     --shot 10 \
#     --dataset 'EuroSATallBands' \
#     --model_name "RN50x16" \
#     --img_size 384 \
#     --root $root

# python eurosat_fs.py \
#     --shot 10 \
#     --dataset 'EuroSATallBands' \
#     --model_name "RN50x64" \
#     --model_name_save "RN50x64" \
#     --img_size 448 \
#     --root $root

# python eurosat_fs.py \
#     --shot 10 \
#     --dataset 'EuroSATallBands' \
#     --model_name "ViT-B/32" \
#     --root $root

# python eurosat_fs.py \
#     --shot 10 \
#     --dataset 'EuroSATallBands' \
#     --model_name "ViT-B/16" \
#     --root $root

