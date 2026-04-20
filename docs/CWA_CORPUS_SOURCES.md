# CWA 官網語料總覽 — 台灣氣象主權 AI

本文件彙整中央氣象署（cwa.gov.tw）及相關官網**所有可作為語料**的來源，供台灣氣象主權 AI 訓練使用。依 API／網頁／檔案等型式分類。

---

## 一、開放資料 API（opendata.cwa.gov.tw）

需申請授權碼：https://opendata.cwa.gov.tw/user/authkey

### 預報類（F-C、F-D、F-A）

| 資料集 ID | 說明 | 語料價值 | 專案狀態 |
|-----------|------|----------|----------|
| **F-C0032-001** | 今明 36 小時天氣預報 | 高 | ✅ 已實作 |
| **F-C0032-003_006** | 一週天氣預報 | 高 | ✅ 已實作 |
| **F-C0044-001** | 氣象報告天氣概況 | 高 | ✅ 已實作 |
| **F-D0047-001~093** | 鄉鎮天氣預報（22 縣市 × 3 天／1 週等） | 高 | 未實作 |
| **F-A0021-001** | 潮汐預報（未來 1 個月） | 中 | ✅ 已實作 |
| **F-A0085-002~005** | 健康氣象（冷傷害、溫差提醒等） | 中 | ✅ 已實作 |

### 天氣警特報（W-C）

| 資料集 ID | 說明 | 語料價值 | 專案狀態 |
|-----------|------|----------|----------|
| **W-C0033-001** | 各縣市天氣警特報情形 | 高 | ✅ 已實作 |
| **W-C0033-002** | 警特報內容及影響區域 | 高 | ✅ 已實作 |
| **W-C0033-003** | 豪大雨特報 | 高 | ✅ 已實作 |
| **W-C0033-004** | 低溫特報 | 高 | ✅ 已實作 |
| **W-C0033-005** | 高溫資訊 | 高 | ✅ 已實作 |
| **W-C0034-001** | 颱風警報 | 高 | ✅ 已實作 |
| **W-C0034-005** | 熱帶氣旋路徑 | 中 | ✅ 已實作 |

### 觀測類（O-A、O-B）

| 資料集 ID | 說明 | 語料價值 | 專案狀態 |
|-----------|------|----------|----------|
| O-A0001-001 | 全測站逐時氣象資料 | 中（數值為主） | 未實作 |
| O-A0002-001 | 雨量資料 | 中 | 未實作 |
| O-A0003-001 | 10 分鐘綜觀氣象資料 | 中 | 未實作 |
| O-A0005-001 | 紫外線指數每日最大值 | 低 | 未實作 |
| O-A0006-002 | 臭氧總量觀測（台北站） | 低 | 未實作 |
| O-B0075-001 | 48 小時海況監測 | 中 | 未實作 |
| O-B0075-002 | 30 天海況監測 | 中 | 未實作 |

### 地震海嘯（E-A）

| 資料集 ID | 說明 | 語料價值 | 專案狀態 |
|-----------|------|----------|----------|
| E-A0014-001 | 海嘯資訊資料 | 高 | ✅ 已實作 |
| E-A0015-001 | 顯著有感地震報告 | 高 | ✅ 已實作 |
| E-A0015-002 | 顯著有感地震報告（英文） | 中 | 未實作 |
| E-A0016-001 | 小區域有感地震報告 | 高 | ✅ 已實作 |
| E-A0016-002 | 小區域有感地震報告（英文） | 中 | 未實作 |

### 氣候類（C-B）

| 資料集 ID | 說明 | 語料價值 | 專案狀態 |
|-----------|------|----------|----------|
| C-B0024-001 | 30 天地面測站觀測 | 中 | 未實作 |
| C-B0025-001 | 每日雨量 | 中 | 未實作 |
| C-B0027-001 | 月平均 | 中 | 未實作 |
| C-B0074-001/002 | 氣象測站基本資料 | 低 | 未實作 |

### 天文類（A-B）

| 資料集 ID | 說明 | 語料價值 | 專案狀態 |
|-----------|------|----------|----------|
| A-B0062-001 | 日出日沒時刻（全臺年度逐日） | 中 | ✅ 已實作 |
| A-B0063-001 | 月出月沒時刻（全臺年度逐日） | 中 | ✅ 已實作 |

### 健康氣象（M-A）

| 資料集 ID | 說明 | 語料價值 | 專案狀態 |
|-----------|------|----------|----------|
| M-A0085-001 | 熱傷害指數及警示 | 中 | 未實作 |

---

## 二、官網知識與天文（文字／QA 語料）

來源：https://www.cwa.gov.tw/V8/C/sitemap.html

### 氣象百科（8-1）

| 來源 | 網址 | 語料價值 | 專案狀態 |
|------|------|----------|----------|
| 氣象常識 | /V8/C/K/Encyclopedia/nous/ | 高 | ✅ 已實作 |
| 氣象儀器 | /V8/C/K/Encyclopedia/inst/ | 高 | ✅ 已實作 |
| 颱風百問 | /V8/C/K/Encyclopedia/typhoon/ | 高 | ✅ 已實作 |
| 地震百問 | scweb.cwa.gov.tw/zh-tw/guidance/faq/ | 高 | 未實作（另站） |
| 天文百問 | /V8/C/K/Encyclopedia/astronomy/ | 高 | ✅ 已實作 |
| 氣候百問 | /V8/C/K/Encyclopedia/climate/ | 高 | ✅ 已實作 |
| 海象百問 | /V8/C/K/Encyclopedia/sea/ | 高 | ✅ 已實作 |
| 氣候變遷問答 | /V8/C/K/Qa/ | 高 | ✅ 已實作 |
| 氣候講堂 | /V8/C/C/Knowledge/knowledge_5.html | 中 | 未實作 |

### 天文／星象

| 來源 | 網址 | 語料價值 | 專案狀態 |
|------|------|----------|----------|
| 每月星象 | /V8/C/K/astronomy_month.html | 高 | 未實作 |
| 每日天文 | /V8/C/K/astronomy_day.html | 高 | 未實作 |

### 其他知識

| 來源 | 網址 | 語料價值 | 專案狀態 |
|------|------|----------|----------|
| 數位科普 | pweb.cwa.gov.tw/PopularScience/ | 高 | ✅ 已實作 |
| 常見問答 | /V8/C/K/CommonFaq/ | 高 | ✅ 已實作 |
| 討論話題 | /V8/C/K/hottopic.html | 高 | 未實作 |
| 有聲書專區 | /V8/C/K/audiobook.html | 中 | 未實作 |
| 雙語詞彙 | /V8/C/K/bilingual_glossary.html | 中 | 未實作 |
| 衛星雲圖精選 | /V8/C/K/Sat_Topic.html | 低（圖為主） | 未實作 |

---

## 三、官網文字報告（需爬取或手動）

### 天氣概況與預報

| 來源 | 網址 | 語料價值 |
|------|------|----------|
| 天氣概況 | /V8/C/W/index.html | 高 |
| 1 週預報 | /V8/C/W/week.html | 高 |
| 相關 PDF | /V8/C/W/pdf.html | 高 |

### 警特報與颱風

| 來源 | 網址 | 語料價值 |
|------|------|----------|
| 天氣警特報 | /V8/C/P/Warning/FIFOWS.html | 高 |
| 高溫資訊 | /V8/C/P/Warning/W29.html | 高 |
| 颱風警報 | /V8/C/P/Typhoon/TY_WARN.html | 高 |
| 颱風消息 | /V8/C/P/Typhoon/TY_NEWS.html | 高 |
| 颱風資料庫 | rdc28.cwa.gov.tw/TDB/ | 高 |

### 農業／生活氣象

| 來源 | 網址 | 語料價值 |
|------|------|----------|
| １週農業氣象 | /V8/C/L/agriculture.html | 高 |
| 氣象旬報 PDF | /V8/C/L/agri_pdf.html | 高 |
| 休閒旅遊預報 | /V8/C/L/（單車、登山、觀星等） | 中 |

### 氣候／海象

| 來源 | 網址 | 語料價值 |
|------|------|----------|
| 短期氣候預測 | /V8/C/C/Forecast/ | 高 |
| 臺灣氣候 | /V8/C/C/Taiwan/ | 高 |
| 氣候變遷 | /V8/C/C/Change/ | 高 |
| 瞭解氣候 | /V8/C/C/Knowledge/ | 高 |
| 潮汐預報 | /V8/C/M/tide.html | 中 |

---

## 四、資料與出版

| 來源 | 網址 | 語料價值 | 專案狀態 |
|------|------|----------|----------|
| 研究出版與年報 | /V8/C/D/publication.html | 高 | ✅ 已實作（列表） |
| 天文資料下載 | /V8/C/D/astronomy_data.html | 中 | 未實作 |
| 氣象產品目錄總集 | /V8/C/D/Data_catalog.html | 中 | 未實作 |

---

## 五、官方社群（需 API 或授權）

| 來源 | 網址 | 語料價值 | 專案狀態 |
|------|------|----------|----------|
| 報天氣（Facebook） | facebook.com/cwa.weather | 高 | ✅ 已實作（需 FB token） |
| 報天氣／報地震／報天文（YouTube） | youtube.com/user/cwbwebtv | 高 | 未實作 |
| RSS | /V8/C/S/eservice/rss.html | 中 | 未實作 |

---

## 六、相關子站（CWA 轄下）

| 子站 | 網址 | 語料價值 |
|------|------|----------|
| 地震測報中心 | scweb.cwa.gov.tw | 高（地震百問、地震報告等） |
| 氣候服務入口網 | climate.cwa.gov.tw | 高 |
| 海象環境資訊平台 | ocean.cwa.gov.tw | 高 |
| 劇烈天氣監測 | qpeplus.cwa.gov.tw | 高 |
| 農業氣象觀測網 | agr.cwa.gov.tw | 中 |
| 健康氣象 | crowa.cwa.gov.tw/HealthWeather/ | 中 |
| 氣象隨選平台 | wede.cwa.gov.tw | 中 |
| 南區氣象服務 | south.cwa.gov.tw | 中 |

---

## 七、專案已實作摘要

| 類別 | 腳本 | 輸出 |
|------|------|------|
| 天氣概況 | fetch_weather_summary.py | official/daily/ |
| 一週預報 | fetch_weekly_forecast.py | official/daily/ |
| 警特報 | fetch_alerts.py | official/alerts/ |
| 額外 API（警特報、地震、潮汐、天文等） | fetch_cwa_extra_api.py | official/alerts/、official/daily/ |
| 官網知識與天文 | fetch_cwa_knowledge.py | official/knowledge/ |
| 數位科普 | fetch_cwa_popular_science.py | official/knowledge/ |
| CWA 出版品 | fetch_cwa_publications.py | academic/tech_reports/ |
| 政府開放平臺 | fetch_data_gov_tw_metadata.py | academic/tech_reports/ |
| PTT 大氣板 | fetch_ptt_ty_research.py | social/ |
| 報天氣 FB | fetch_facebook_cwa.py | social/ |

---

## 八、建議擴充優先順序

1. **高優先**：~~W-C0033 警特報 API~~、F-D0047 鄉鎮預報、每月星象、每日天文、~~數位科普~~
2. **中優先**：地震百問、E-A 地震報告、短期氣候預測、潮汐預報
3. **低優先**：O-A 觀測（數值轉文字）、雙語詞彙、RSS

---

## 參考連結

- 官網導覽：https://www.cwa.gov.tw/V8/C/sitemap.html
- 開放資料平台：https://opendata.cwa.gov.tw/
- API 說明：https://opendata.cwa.gov.tw/dist/opendata-swagger.html
- 授權碼申請：https://opendata.cwa.gov.tw/user/authkey
