# 图表规格

`scripts/stock_chart.py` 把 JSON 规格渲染成与 `docs/WikiImage/` 同风格的 SVG：宽 900、白底圆角、系统字体、标题 20px、底部灰色注释条。坐标全部由数据算出。

```bash
python3 -I .claude/skills/stock-history-pricing/scripts/stock_chart.py spec.json docs/WikiImage/<topic>_<aspect>.svg
rsvg-convert -z 1.5 docs/WikiImage/<topic>_<aspect>.svg -o /tmp/<aspect>.png   # 然后查看 PNG
```

脚本写完会做 XML 校验并输出 `ok <路径>`；`WARN` 行表示文字可能越出画布或列宽超限，需调整后重跑。示例见 `examples/`：`pe_profit.json`（PE + 利润）、`forecast.json`（研报预测轨迹）、`expectation.json`（预期差时间线）。

## 颜色约定

色名取 `blue`、`teal`、`amber`、`grape`、`red`、`gray`，每个色名对应描边、文字、浅底三档。全 Wiki 保持同一含义：

| 色名 | 用途 |
| --- | --- |
| blue | 价格、PE、机构研报 |
| teal | 利润、基本面 |
| amber | 事件、高亮年份、关注点切换 |
| grape | 散户与媒体 |
| red | 风险、过热阶段、“当时低估” |
| gray | 背景、低谷阶段、实际值图例 |

## 时间语义

日期可写 `2021-07-07`、`2021-07`、`2021Q3`、`2021H1`、`2021` 或数字。

- **数据点**画在所属期间的中点：`"2021"` 是年中，`"2021Q3"` 是三季度中间。年度柱、年度 PE 因此落在年份格子中央。
- **边界**（`x_range`、`x_ticks`、`bands`、`spans` 默认以外）取期间起点：`"2022"` 即 2022-01-01。
- 年度图用 `"x_labels": "center"`，年份标签居中于格子；日度、季度图用默认 `"start"`，标签贴在刻度右侧。

## 一、`panels`：多面板时间轴

上下堆叠若干面板，共用一条时间轴。适合 PE 与利润对照、价格与单季利润的领先滞后、研报预测轨迹。

顶层字段：

| 字段 | 说明 |
| --- | --- |
| `type` | `"panels"`（默认） |
| `title` / `subtitle` | 标题与副标题，副标题写口径和来源 |
| `x_range` | `[起, 止]`，按边界解析，如 `["2020", "2026"]` 覆盖 2020—2025 六整年 |
| `x_ticks` / `x_tick_labels` | 刻度年份与可选的自定义标签；默认取 `x_range` 内的整年 |
| `x_labels` | `"center"` 或 `"start"` |
| `bands` | `[{from, to, label, color}]` 阶段底色，贯穿所有面板，标签在面板上方 |
| `events` | `[{date, label, color, anchor, dy}]` 竖向虚线，标签在第一面板顶部；`anchor: "end"` 把标签放到线左侧，`dy` 下移避让 |
| `spans` | `[{from, to, label, color, after_panel, edge}]` 两个面板之间的括号，用来标“股价领先利润几个月”；`after_panel` 为面板序号，`edge: true` 时按边界解析日期 |
| `panels` | 面板数组，见下 |
| `legend` | `[{name, color, kind}]`，`kind` 取 `line`、`dash`、`bar`、`box`、`square` |
| `note` | 字符串数组，每项一段，自动折行；写结论和来源 |

面板通用字段：

| 字段 | 说明 |
| --- | --- |
| `kind` | `line`、`bar`、`forecast` |
| `label` / `label_color` | 面板左上角标题，写指标与单位 |
| `height` / `gap` | 面板高度（默认 200）与到下一面板的间距（默认 36；有 spans 时调大） |
| `y_range` / `y_ticks` / `y_tick_count` | 纵轴范围与刻度；不填时自动外扩到整刻度 |
| `y_suffix` | 刻度后缀，如 `"×"`、`"%"` |
| `zero` | 自动范围是否包含 0，默认 `true`；PE、价格面板可设 `false` |
| `ranges` | `[{from, to, color, label}]` 横向区间底色，如全期 PE 区间 |
| `hlines` | `[{value, color, label, dx, dy, anchor}]` 水平参考线，如长期中枢 |
| `marks` | `[{date, value, label, sub, color, dx, dy, anchor}]` 标注关键点（极值、拐点） |

### `line` 面板

`series: [{name, color, points, dashed, dots, width}]`，`points` 为 `[[日期, 值], ...]`；值写 `null` 会断开折线，用于停牌、亏损期 PE 无意义等缺口。

### `bar` 面板

只画第一个 series。`highlight: {"2025": "amber"}` 给指定期间换色并加粗数值；`value_labels` 为 `true`（全部标数）、`false` 或期间列表；`bar_width` 以年为单位，默认相邻间隔的 55%。负值向下画，适合亏损年份。

### `forecast` 面板

一条折线代表**同一目标年度**的历次预测，横轴是报告发布日：

```json
{"kind": "forecast", "height": 300, "targets": [
  {"name": "2022", "color": "teal",
   "forecasts": [["2021-01-05", 153, "2022E 153"], ["2021-07-08", 240, "2022E 240"]],
   "actual": ["2023-03-30", 1096, "2022 实际 1,096", {"dx": 0, "dy": -14, "anchor": "middle"}]}
]}
```

`actual` 画在披露日，用空心方块和虚线连到最后一次预测。线越陡，说明研报越是在追着事实修正。配合 `events` 标出股价顶点，就能直接看出“股价见顶时预测还在哪里”。

## 二、`expectation`：预期差时间线

竖向时间线，每行一个节点，列出事件、机构与散户的看法，最后一列是事后判断的预期差。

```json
{"type": "expectation", "title": "...", "subtitle": "...",
 "rows": [{"date": "2021-07-07", "date_sub": "股价见顶",
           "event": "半年报预告 370.9 亿元", "inst": "...", "retail": "...",
           "gap": "关注点切换：开始交易 2022 年回落", "gap_dir": "shift"}],
 "note": ["..."]}
```

- `gap_dir` 取 `under`（红，当时低估、后来上修）、`over`（青，当时高估、后来下修）、`shift`（琥珀，关注点切换）、`match`（灰，基本符合）。时间轴圆点与预期差格同色，图例只列用到的方向。
- `gap` 第一行加粗，写成“方向：一句证据”，如“仍在低估：实际为预测的 5 倍”。
- 单元格自动折行，行高随最长的格子变化。每格控制在 40 字以内，6—8 行为宜；更多节点拆成两张图。
- 默认四列：`event` 200、`inst` 190、`retail` 170、`gap` 132。可用 `columns: [{key, title, width, color, gap}]` 改列（如增加“股价 / PE”列），列宽与列间距（每列 6）合计不超过 720。
- `date_title` 改左侧时间列标题。

## 标签避让

数据点标签、`marks`、`hlines`、预测点和实际值都接受 `{dx, dy, anchor}`：`dx`、`dy` 为像素偏移，`anchor` 取 `start`、`middle`、`end`。相邻点标签重叠时，一个放上（`dy` 负）、一个放下（`dy` 正）；靠右边界的标签用 `anchor: "end"`。先跑一遍看 PNG，再逐个微调，不要一开始就给所有点写偏移。

## 选图与写图

- 一张图一个问题，标题直接写结论（“估值先行，利润后到”），副标题写口径、单位、来源。
- 图中每个数字都要能在正文表格里找到；示例数据只用于演示，不能出现在新标的的页面里。
- 年度平均 PE 缺日度数据时只画年末值，并在 `note` 里说明缺项。
- 事后才知道的信息（实际值、后来的高点）用空心方块、虚线或注释区分，不要和当时可见的数据混成一条线。
- 文件命名 `<topic>_<aspect>.svg`，如 `gyfl_pe_profit.svg`、`cosco_shipping_cycle_forecast.svg`。
