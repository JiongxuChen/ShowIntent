# 推送到GitHub的main分支

以下命令在本仓库根目录执行。更新直接推送到远端main。

## 提交本地更新

```bash
git switch main
git status --short
git diff --stat
git add -A
git diff --cached --stat
git diff --cached --name-only
git commit -m "Refine presentation intent and update evaluation reports"
```

提交前确认清单中没有原视频、模型权重、虚拟环境或个人路径。如果没有待提交内容，跳过commit步骤。如果切换main时提示本地改动会被覆盖，先处理这些改动，不要使用强制切换。

## 合并远端更新并推送

```bash
git fetch origin
git merge --no-edit origin/main
git push origin main
```

只有merge成功后才执行push。远端没有额外更新时，merge会提示已经是最新状态；存在可自动合并的更新时，Git会保留双方提交。

## 如果出现合并冲突

查看冲突文件：

```bash
git status
```

逐个编辑冲突文件，删除冲突标记并保留正确的实现。视觉模型、B逻辑发生冲突时，需要核对接口与评价结果，不能直接用一方的整个文件覆盖另一方。

解决后执行：

```bash
git add -A
git diff --cached --check
python -m pytest -q
git commit --no-edit
git push origin main
```

如果暂时不准备解决冲突，可以用`git merge --abort`撤销本次合并，本地已完成的提交仍会保留。

如果push再次提示fetch first，说明其他成员在此期间又更新了main，重新执行fetch和merge，再推送。不要使用强制推送覆盖远端。
