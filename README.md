# NBER 公共经济与管理雷达

这个版本主要做两件事：

1. 自动推荐与公共经济、公共管理相关的 NBER Working Paper；
2. 支持中文关键词检索，并按相关程度排序。

## 启动

需要 Python 3.10 或更高版本。

```bash
python -m venv .venv
```

Windows：

```bash
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

macOS / Linux：

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## 这次修复的重点

- 推荐范围改为公共经济、公共管理及相关政策研究；
- 修复 `AI` 误命中 `training`、`fair` 等单词的问题；
- 搜索改为先搜索全库、再评分、最后截取结果；
- 修复相关度评分失真；
- 修复缓存数量被第一次选择锁死的问题；
- “刷新”现在会真正清除论文和数据缓存；
- 修复常见英文引号乱码；
- 网络失败时会尽量使用本地备份，并显示明确提示；
- “忽略”后的论文不再推荐；“收藏”会轻微影响后续排序；
- 搜索后不再自动批量调用 AI，避免页面卡顿和大量消耗额度；
- API Key 只保存在当前网页会话，不再写入配置文件，也不再支持通过网址传入。

## AI 中文摘要

在网页左侧填写 API Key。Key 只保存在当前页面会话中。

模型地址和模型名可以保存，但配置文件中不会保存 Key。也可以通过环境变量提供：

```bash
NBER_RADAR_API_KEY=你的Key
```

之前使用过的旧 Key 如果已经写进文件或分享给别人，应当到服务平台删除并重新生成。

## 命令行生成周报

```bash
python cli.py --max-papers 50 --min-score 25
```

强制刷新：

```bash
python cli.py --refresh
```

## 本地备份数据

项目中的 `papers.json`、`ref.tsv`、`prog.tsv` 只用于网络失败时临时备用。正常联网时，程序会优先获取 NBER 最新数据。

## 自检

```bash
python -m unittest -v test_nber_radar.py
```

Windows 用户也可以直接双击 `START.bat` 启动。

## 0.4 检索逻辑

- 拆词改为"概念组"（组内或、组间且），而非扁平列表，解决"数字基础设施"等多词命中失序的问题。
- 搜索得分只看查询本身（完全去掉公共经济相关度的 0.35 倍混入），避免无关论文拿满分。
- 新增不认识的词（uncovered）检测和提示，指导用户用 API Key 启用大模型拆词。
- 建全库检索索引并缓存，向量化程度表生成，正则编译 lru_cache，单次搜索从 18–68 秒降至 ≤3 秒。

## 0.5 中文译文

- 填写 API Key 后，搜索结果前几篇（默认 5 篇，可调）自动翻译，其余可一键翻译。每篇生成中文题目、完整摘要译文和一句话（只写摘要里明确有的内容）。
- 译文永久缓存在本机 `nber_radar.sqlite`，同一篇不重复翻译。
- 推荐模型 `glm-4-flash-250414`（智谱免费、较快）。高峰期 `glm-4.7-flash` 常被限流，限流时会自动重试并改用备选模型，失败原因会显示在页面上。
- 翻译时附带术语表（词典和本次检索概念），避免 patient capital 被译成”患者资本”之类的误译。
- 英文检索词自动兼容常见复数（intention/intentions）。

## 0.6 年份与分类筛选

- 搜索时可按年份范围（基于发表日期）和 NBER 研究项目分类（多选）筛选。
- 侧栏新增年份范围滑块（默认全选）和分类多选框（默认不限），仅限于全库搜索，推荐路径不受影响。
- 筛选条件在结果页显示，清晰表示生效的条件。
- 结果可导出为 Markdown、CSV、RIS（Zotero/EndNote 通用）、BibTeX 格式。导出时自动正确处理作者字段，支持分号或逗号分隔的两种标准格式。

## 0.6.2 保存检索主题

- 搜索结果上方新增"保存为主题"功能，输入名称后可将当前概念组、年份范围、分类筛选保存为主题。同名主题会覆盖。
- 侧栏"我的主题" expander 列出全部已保存主题，每个主题可一键"运行"（不调用大模型，直接用保存的参数搜索）或删除。
- 为周推功能预留 `pushed_papers` 表，记录已推送论文 ID 以便去重（本版本仅建表和函数，下阶段实现推送）。

## 0.6.3 检索词可编辑

- 搜索结果展示后，点开"调整检索词"可编辑每个概念组的英文词、删除不需要的组、新增概念。
- 编辑完后点"按修改后的词重搜"即刻重新搜索，不调用大模型，直接用编辑后的词。年份/分类筛选条件保留。
- 支持逗号分隔多词，自动处理去重、小写归一化，单组最多 12 词，最多 5 个组。

## 0.6.4 每周微信推送

运行 `weekly_push.py` 脚本推送保存的主题最近一周新发布的论文到微信。消息约 3000 字符自动拆分，超长行也会递归拆分确保每条消息不超限。

```bash
python weekly_push.py                    # 推送所有主题
python weekly_push.py --dry-run          # 只打印，不发送
python weekly_push.py --days 14          # 推送最近两周
python weekly_push.py --topic 主题名     # 只推送指定主题
python weekly_push.py --no-refresh       # 不刷新 NBER 数据
```

设置定时任务（macOS）：

```bash
cp com.nberradar.weekly.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.nberradar.weekly.plist
```

每周一 09:07 自动运行，日志写到 `weekly_push.log`。

