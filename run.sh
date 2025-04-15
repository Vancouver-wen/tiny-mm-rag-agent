export CUDA_VISIBLE_DEVICES=3

python ./main.py -t build -c config/deepseek_config.yaml -p data/raw_data/abc_cn_20250315.json
python ./main.py -t search -c config/deepseek_config.yaml
