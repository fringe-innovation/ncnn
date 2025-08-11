#!/usr/bin/env bash

# -------------------------
# 配置部分
# -------------------------
# 本地源目录（不带末尾斜杠以避免意外）
SRC_DIR="."

# 远程目标信息
DEST_USER="szbaijie"
DEST_IP="192.168.31.5"
DEST_PORT=22
DEST_DIR="/mnt/drive/ncnn_build/ncnn_a133"

# 排除模式数组：忽略所有 build/ 和 install/ 目录及其子目录
EXCLUDES=(
  "--exclude=build/"
  "--exclude=install/"
    "--exclude=.git/"
    "--exclude=.vscode/"
    "--exclude=__pycache__/"
)

# -------------------------
# 同步命令
# -------------------------
rsync -avz \
  "${EXCLUDES[@]}" \
  -e "ssh -p ${DEST_PORT}" \
  "${SRC_DIR}/" \
  "${DEST_USER}@${DEST_IP}:${DEST_DIR}/"
