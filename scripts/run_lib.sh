#!/bin/bash

retry_command() {
    # 获取命令参数
    local command="$1"

    # 定义重试次数和等待时间
    local max_attempts=100
    local wait_time=10

    # 初始化计数器
    local attempt=0

    while [ $attempt -lt $max_attempts ]
    do
        if eval $command; then
            # 如果命令执行成功，输出信息并退出循环
            echo "Command executed successfully."
            break
        else
            # 如果命令执行失败，则增加计数器并等待一段时间后重试
            attempt=$((attempt+1))
            echo "Command failed $attempt, Retrying in $wait_time seconds..."
            sleep $wait_time
            svn cleanup
        fi
    done

    # 如果达到最大重试次数，输出错误信息
    if [ $attempt -eq $max_attempts ]; then
        echo "Command failed after $max_attempts attempts."
    fi
}
# cd /home/sherry/cyclephi-git/src/lib/linux_centos7_x86_64
# svn cleanup
# 调用函数，传入要执行的命令作为参数
# retry_command "svn up"
retry_command "./scripts/run_cyclesphi33_cpu.sh"
