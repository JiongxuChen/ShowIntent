# 更新GitHub

在本仓库根目录执行。先检查文件及指标，再提交到当前分支；推送前确认远端和分支。

```bash
git status --short
git diff --stat
python -m pytest -q
python -m scripts.run_refined_b --clips E04 --out outputs/release_check_e04
git add README.md .gitignore requirements*.txt pytest.ini src scripts configs tests evaluation docs outputs release_manifest.json ShowIntent_*.md
git diff --cached --stat
git status --short
git commit -m "Update offline five-method evaluation and delayed streaming prototype"
git remote -v
git branch --show-current
git push
```

仓库不发布原视频、权重、虚拟环境或新生成媒体。`outputs`中只对本次选定证据目录开放跟踪，新增运行默认忽略。提交前检查`git diff --cached --name-only`，确认没有个人路径、凭据、媒体或模型文件。项目文档只使用相对路径。
