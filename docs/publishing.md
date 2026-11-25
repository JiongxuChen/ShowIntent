# 上传GitHub：完整操作步骤

本文从已经整理好的本地Git仓库开始。发布内容为离线A、B、简易C、D、标注、缓存、预测和评价；原视频、模型权重、虚拟环境和临时运行产物不提交。数据用途为水果组用于开发，书籍、水杯、饮料用于评价，报告按场景组织。

以下步骤由操作者执行；目录整理不会自动commit、push或合并。

## 1. 进入仓库并确认位置

在终端进入准备上传的ShowIntent仓库根目录，即同时包含.git、README.md、src、scripts、evaluation和outputs的目录，不是存放原视频的工作目录。

```bash
cd /path/to/ShowIntent
git rev-parse --show-toplevel
git branch --show-current
git remote -v
git status --short
```

将/path/to/ShowIntent替换为实际目录。当前origin应指向`https://github.com/JiongxuChen/ShowIntent.git`；SSH形式`git@github.com:JiongxuChen/ShowIntent.git`也可以。

如果没有origin，添加：

```bash
git remote add origin https://github.com/JiongxuChen/ShowIntent.git
```

如果origin指错仓库，先确认目标，再修改：

```bash
git remote set-url origin https://github.com/JiongxuChen/ShowIntent.git
```

## 2. 检查本次准备发布的内容

```bash
git diff --stat
git diff --check
git status --short --untracked-files=all
```

M表示修改、D表示删除、??表示新文件。删除已被合并的旧文档和工具属于这次整理的一部分，需要一起提交。检查时应看到src、scripts、configs、docs、evaluation、tests以及精选outputs，而不应看到Example_Data目录、models或虚拟环境。

outputs允许提交的目录为cache、results、benchmarks；它们分别是A/D缓存、正式B结果及模型对比。它们不是重复临时文件。新重放目录默认被忽略。

不要使用git add -f把被忽略的视频、权重或环境强行加入。

## 3. 验证可复现入口

正式结果整理时已核对：277项测试通过、9项跳过；22段三模式评分一致。发布前如代码有新增修改，可再运行：

```bash
python -m pip install -r requirements-b-dev.txt
python -m pytest -q
python -m scripts.run_refined_b --out outputs/publish_check
```

缓存B重放不需GPU或原视频。outputs/publish_check必须是新目录，生成的临时结果不提交。完整A/D需要额外安装README所述依赖、模型及视频。

如修改过发布文件，需同步更新release_manifest.json；它记录实际文件SHA-256，文档不应宣称旧哈希仍代表修改后的文件。

## 4. 确认使用main分支

本项目直接提交并推送到main，不另建发布分支或发起Pull Request。

```bash
git switch main
git branch --show-current
```

若已经在main，git switch不会改变工作文件。若切换时提示本地改动冲突，先检查分支与文件，不使用强制切换丢弃改动。

## 5. 暂存新增、修改和删除

在仓库根目录执行：

```bash
git add -A
git diff --cached --stat
git diff --cached --check
git diff --cached --name-status
git status --short
```

使用git add -A是为了同时记录旧文件删除。正常情况下，准备发布的新增文件也会进入暂存区；.gitignore继续排除视频、权重与临时输出。

如发现某个文件不应提交，取消暂存即可，不会删除本地文件：

```bash
git restore --staged path/to/file
```

可用以下检查确认暂存区没有媒体、权重或超过单文件100MiB的文件：

```bash
python - <<'PY_CHECK'
import subprocess
from pathlib import Path
names = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z']).decode().split('\0')
for name in filter(None, names):
    path = Path(name)
    size = int(subprocess.check_output(['git', 'cat-file', '-s', ':' + name]))
    if path.suffix.lower() in {'.mp4', '.mov', '.avi', '.wav', '.pt', '.onnx', '.zip'} or size >= 100 * 1024 * 1024:
        raise SystemExit('请检查暂存文件：' + name)
print('暂存文件检查通过')
PY_CHECK
```

## 6. 创建本地提交

```bash
git commit -m "Publish offline visual-audio presentation intent pipeline and evaluation"
git log -1 --oneline
git status --short
```

若提示未设置作者身份，填写自己的真实GitHub提交身份，再重新执行commit：

```bash
git config user.name "Your Name"
git config user.email "your-email@example.com"
```

上述配置仅作用于当前仓库。提交后，git status应没有遗漏的待发布文件；被忽略的临时重放目录不会显示。

## 7. 同步远程main并推送

本地commit完成后，先获取远程状态：

```bash
git fetch origin
git log --oneline --left-right HEAD...origin/main
```

若远程main在本次整理期间新增了提交，先将自己的提交重放到远程main之上：

```bash
git rebase origin/main
```

发生冲突时，打开git status列出的文件，核对双方内容，解决冲突标记后执行：

```bash
git add path/to/resolved-file
git rebase --continue
```

直到rebase完成。若需要取消此次rebase，执行`git rebase --abort`，本地提交仍保留。解决冲突后应重新检查受影响内容；不使用强制推送覆盖队友提交。

确认在main且提交完整后直接推送：

```bash
git push -u origin main
```

HTTPS认证使用GitHub支持的凭据方式，不在命令或文档中写入令牌。若提示权限不足，确认账号有写权限；若仓库保护规则禁止直推main，则需要仓库管理员调整规则，不能用强制推送绕过。

## 8. 核对GitHub上的main

推送完成后检查：

```bash
git status --short
git log -1 --oneline
git rev-parse HEAD
git ls-remote origin refs/heads/main
```

本地HEAD与远程main提交哈希应一致。打开GitHub仓库并选择main，确认：

- README展示离线A/B/简易C/D、运行命令和四视觉模型入口。
- docs包含A.md、B.md、C.md、D.md、评价、模型比较和逐文件结构说明。
- evaluation按四场景组织；outputs包含cache、results、benchmarks。
- benchmarks下四场景均有YOLO11n/m/l、RT-DETR-L的评价，共16份JSON。
- 未夹带原视频、权重、虚拟环境、临时输出或延时流式内容。

推送到main后即完成本次发布，无需额外创建Pull Request。可在另一目录重新克隆仓库，按README执行缓存B重放，检查已提交内容是否完整。
