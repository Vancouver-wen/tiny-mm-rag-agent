export CUDA_VISIBLE_DEVICES=3

# python script/tiny_rag.py -t build -c config/qwen2_config.json -p data/raw_data/wikipedia-cn-20230720-filtered.json
# python script/tiny_rag.py -t search -c config/qwen2_config.json

python ./tiny_rag.py -t build -c config/deepseek_config.yaml -p data/raw_data/abc_cn_20250315.json
python ./tiny_rag.py -t search -c config/deepseek_config.yaml
