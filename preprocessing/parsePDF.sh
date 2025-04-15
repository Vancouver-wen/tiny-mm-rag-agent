#!/bin/bash

# 设置CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="3"

# magic-pdf \
#     -p /data/wzh_fd/workspace/tinyRAG/data/dfcf/reports-list/0 \
#     -o /data/wzh_fd/workspace/tinyRAG/data/dfcf/outputs \
#     -m ocr


# 定义输入文件夹和输出文件夹
base_input_folder="/data/wzh_fd/workspace/tinyRAG/data/dfcf/reports-list"
output_folder="/data/wzh_fd/workspace/tinyRAG/data/dfcf/outputs"

# 确保输出文件夹存在
mkdir -p "$output_folder"

# 总文件夹数量
total_folders=25
current_folder=0
total_time=0

# 遍历从 0 到 24 的所有文件夹
for folder_num in {0..24}; do
    # 构造当前文件夹路径
    current_input_folder="$base_input_folder/$folder_num"

    # 检查当前文件夹是否存在
    if [ -d "$current_input_folder" ]; then
        # 更新进度条
        current_folder=$((current_folder + 1))
        progress=$((current_folder * 100 / total_folders))
        echo -ne "Processing: [${progress}%] Folder $folder_num\r"

        # 记录当前文件夹的开始时间
        start_time=$(date +%s)

        # 调用 magic-pdf 处理当前文件夹
        magic-pdf \
            -p "$current_input_folder" \
            -o "$output_folder" \
            -m ocr

        # 记录当前文件夹的结束时间
        end_time=$(date +%s)
        elapsed_time=$((end_time - start_time))

        # 更新总处理时间和已处理文件夹数量
        total_time=$((total_time + elapsed_time))

        # 计算平均处理时间和剩余时间
        if [ $current_folder -gt 0 ]; then
            avg_time_per_folder=$((total_time / current_folder))
            remaining_folders=$((total_folders - current_folder))
            remaining_time=$((avg_time_per_folder * remaining_folders))
            hours=$((remaining_time / 3600))
            minutes=$(((remaining_time % 3600) / 60))
            seconds=$((remaining_time % 60))
            echo -ne "Estimated remaining time: ${hours}h ${minutes}m ${seconds}s\r"
        fi
    else
        echo "=> 文件夹 $current_input_folder 不存在，跳过 .."
    fi
done

echo -ne "\n所有文件夹处理完成。"

# # 定义输入文件夹
# input_folder="/data/wzh_fd/workspace/tinyRAG/data/dfcf/reports"
# output_path="/data/wzh_fd/workspace/tinyRAG/data/dfcf/outputs"

# # 获取文件夹中PDF文件的数量
# total_files=$(ls "$input_folder"/*.pdf | wc -l)
# current_file=0
# # 最大重试次数
# max_retries=5

# # 初始化总处理时间和文件计数
# total_time=0
# processed_files=0

# # 遍历输入文件夹中的所有PDF文件
# for pdf_file in "$input_folder"/*.pdf; do
#     # 获取文件名（不带路径）
#     filename=$(basename "$pdf_file")

#     # 初始化重试计数
#     retry_count=0

#     # 更新进度条
#     current_file=$((current_file + 1))
#     progress=$((current_file * 100 / total_files))

#     # 记录当前文件的开始时间
#     start_time=$(date +%s)

#     echo -ne "Processing: [${progress}%] $filename"

#     # 尝试处理PDF文件，最多重试$max_retries次
#     while [ $retry_count -lt $max_retries ]; do
#         # 调用magic-pdf处理PDF文件
#         if magic-pdf -p "$pdf_file" -o "$output_path"; then
#             echo -ne "\nSuccessfully processed $filename\n"
#             break
#         else
#             retry_count=$((retry_count + 1))
#             echo -ne "\nFailed to process $filename, retrying... ($retry_count/$max_retries)\n"
#         fi
#     done

#     # 记录当前文件的结束时间
#     end_time=$(date +%s)
#     elapsed_time=$((end_time - start_time))

#     # 如果文件处理成功，更新总时间和文件计数
#     if [ $retry_count -lt $max_retries ]; then
#         total_time=$((total_time + elapsed_time))
#         processed_files=$((processed_files + 1))
#     fi

#     # 计算平均处理时间和剩余时间
#     if [ $processed_files -gt 0 ]; then
#         avg_time_per_file=$((total_time / processed_files))
#         remaining_files=$((total_files - current_file))
#         remaining_time=$((avg_time_per_file * remaining_files))
#         hours=$((remaining_time / 3600))
#         minutes=$(((remaining_time % 3600) / 60))
#         seconds=$((remaining_time % 60))
#         echo -ne "Estimated remaining time: ${hours}h ${minutes}m ${seconds}s\r"
#     fi

#     echo -ne "\n"
# done

echo "All PDF files have been processed."