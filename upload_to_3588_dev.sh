#!/usr/bin/env bash

# -------------------------
# 配置部分
# -------------------------
# 本地源目录（不带末尾斜杠以避免意外）
SRC_DIR="."

# 远程目标信息
DEST_USER="orangepi"
DEST_IP="110.41.144.65"
DEST_PORT=2093
DEST_DIR="/home/orangepi/rknn_ws/rknn_modified"

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
