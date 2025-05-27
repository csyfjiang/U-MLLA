#!/bin/sh

# 定义要运行的脚本列表
scripts="train_mlla_acdc.sh train_mlla_altas.sh train_mlla_amos.sh train_mlla_word.sh train_mlla_flare22.sh train_mlla_amos_mr.sh"
# train_mlla_flare22.sh
# train_mlla_amos_mr.sh
# train_mlla_btcv.sh
# 显示当前目录
echo "Current directory: $(pwd)"

# 显示脚本文件的详细信息
ls -l train_mlla_*.sh

# 遍历并运行每个脚本
for script in $scripts; do
    echo "Running $script..."
    
    # 检查脚本是否存在且可执行
    if [ -f "$script" ]; then
        if [ -x "$script" ]; then
            echo "Executing $script"
            ./"$script"
        else
            echo "Warning: $script found but not executable. Trying to add execute permission."
            chmod +x "$script"
            if [ -x "$script" ]; then
                echo "Execute permission added. Executing $script"
                ./"$script"
            else
                echo "Failed to add execute permission to $script. Skipping."
            fi
        fi
    else
        echo "Warning: $script not found. Skipping."
    fi
    
    echo "Finished $script"
    echo "------------------------"
done

echo "All scripts have been executed."