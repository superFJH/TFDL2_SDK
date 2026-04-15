#!/bin/bash
#
# sync_to_github.sh — 将内部git仓库同步到GitHub
#
# 用法：
#   首次同步：./sync_to_github.sh <GitHub用户名或组织>
#   后续同步：./sync_to_github.sh
#
# 示例：
#   ./sync_to_github.sh myuser
#   ./sync_to_github.sh MyOrg
#
# 说明：
#   - 仅同步 git ls-files 列出的已追踪文件
#   - 不共享 .git 目录，创建全新的独立仓库
#   - GitHub仓库名：TFDL2_SDK
#

set -euo pipefail

# ===== 配置 =====
REPO_NAME="TFDL2_SDK"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SYNC_DIR="${SCRIPT_DIR}/../${REPO_NAME}_github"

# ===== 参数处理 =====
if [ $# -ge 1 ]; then
    GITHUB_USER="$1"
    GITHUB_REMOTE="git@github.com:${GITHUB_USER}/${REPO_NAME}.git"
    echo ">>> GitHub远程仓库: ${GITHUB_REMOTE}"
else
    # 后续同步时，自动从已有remote获取
    if [ -d "${SYNC_DIR}/.git" ]; then
        GITHUB_REMOTE=$(cd "${SYNC_DIR}" && git remote get-url origin 2>/dev/null || true)
        if [ -z "${GITHUB_REMOTE}" ]; then
            echo "错误：请首次运行时指定GitHub用户名: $0 <GitHub用户名或组织>"
            exit 1
        fi
        echo ">>> 使用已有远程仓库: ${GITHUB_REMOTE}"
    else
        echo "错误：请首次运行时指定GitHub用户名: $0 <GitHub用户名或组织>"
        exit 1
    fi
fi

# ===== 步骤1：创建同步目录 =====
echo ""
echo ">>> [1/5] 准备同步目录: ${SYNC_DIR}"
mkdir -p "${SYNC_DIR}"

# 清空同步目录（保留.git）
if [ -d "${SYNC_DIR}/.git" ]; then
    # 已有git仓库，保留.git，清空其余文件
    find "${SYNC_DIR}" -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +
else
    # 全新目录，清空
    rm -rf "${SYNC_DIR:?}"/*
fi

# ===== 步骤2：复制git追踪的文件 =====
echo ">>> [2/5] 复制git追踪文件..."
cd "${SCRIPT_DIR}"

# 使用 git ls-files -z (null分隔) 获取所有已追踪文件，避免中文路径编码问题
TRACKED_COUNT=0
git ls-files -z | while IFS= read -r -d '' file; do
    dest="${SYNC_DIR}/${file}"
    mkdir -p "$(dirname "${dest}")"
    cp -f "${file}" "${dest}"
    TRACKED_COUNT=$((TRACKED_COUNT + 1))
done
TRACKED_COUNT=$(git ls-files | wc -l)
echo "    已复制 ${TRACKED_COUNT} 个文件"

# ===== 步骤3：初始化git仓库（仅首次） =====
if [ ! -d "${SYNC_DIR}/.git" ]; then
    echo ">>> [3/5] 初始化新git仓库..."
    cd "${SYNC_DIR}"
    git init
    git remote add origin "${GITHUB_REMOTE}"
else
    echo ">>> [3/5] 使用已有git仓库..."
    cd "${SYNC_DIR}"
    # 确保remote正确
    CURRENT_REMOTE=$(git remote get-url origin 2>/dev/null || true)
    if [ "${CURRENT_REMOTE}" != "${GITHUB_REMOTE}" ]; then
        git remote set-url origin "${GITHUB_REMOTE}"
    fi
fi

# ===== 步骤4：提交变更 =====
echo ">>> [4/5] 提交变更..."
git add -A

# 检查是否有变更需要提交
if git diff --cached --quiet; then
    echo "    没有变更需要提交"
else
    # 从源仓库获取最新commit信息作为提交信息
    SOURCE_COMMIT=$(cd "${SCRIPT_DIR}" && git log -1 --oneline)
    git commit -m "sync from internal repo (${SOURCE_COMMIT})"
    echo "    提交完成"
fi

# ===== 步骤5：推送到GitHub =====
echo ">>> [5/5] 推送到GitHub..."
cd "${SYNC_DIR}"

# 检查是否已有远程分支
BRANCH="main"
if git ls-remote --heads origin "${BRANCH}" 2>/dev/null | grep -q "${BRANCH}"; then
    # 远程已有分支，直接push
    git push origin "${BRANCH}"
else
    # 首次推送，设置上游分支
    git branch -M "${BRANCH}"
    git push -u origin "${BRANCH}"
fi

echo ""
echo ">>> 同步完成！"
echo "    仓库地址: ${GITHUB_REMOTE}"
echo "    本地目录: ${SYNC_DIR}"
