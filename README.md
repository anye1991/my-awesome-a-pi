# my-awesome-a-pi

## 项目简介

my-awesome-a-pi 是一套顶级 AI 自动化渗透测试与智能安全分析 API 平台，集成了多模态智能推理、批量任务处理、结构化聚合、超长文本支持、知识增强分析等能力。**已完全移除所有输入长度限制**，适合高强度安全分析、红队自动化、智能安全运维等场景。

---

## 主要功能与特点

- **自动化渗透测试引擎**：支持文本、日志、代码、协议流等多模态输入，智能分析安全隐患。
- **多模型推理/聚合**：内置多种 LLM/ML 模型，自动分片聚合，极限支持超长文本。
- **输入无限制**：彻底移除输入长度限制，可处理万字以上内容。
- **批量与结构化任务**：一行请求可分析多个目标，返回结构化聚合结果。
- **智能决策与辅助分析**：自动决策、命令生成、漏洞分析、资产归类等高级功能。
- **专业接口设计**：参数健壮，异常处理完善，支持 JSON/文本多种返回格式。
- **安全合规**：仅限授权安全测试与研究用途。

---

## 环境依赖与安装

建议使用 Python 3.8 及以上。  
一键安装依赖：

```
pip install -r requirements.txt
```

---

## 启动方法

```
python 32.py
```
默认监听本地 5000 端口。

---

## 主要API端点与详细使用教程

### 1. /api/analyze

- **功能**：智能语义安全分析，自动检测文本/日志/源码等中的安全风险。
- **请求方式**：POST
- **请求参数**：
  - `text`：输入内容（string，无长度限制）

- **请求示例**：

```
curl -X POST http://localhost:5000/api/analyze -H "Content-Type: application/json" -d '{"text": "select * from users where id=1"}'
```

- **返回示例**：

```json
{
  "status": "success",
  "data": {
    "risk_score": 0.86,
    "confidence": 0.91,
    "is_malicious": true,
    "threat_level": "critical",
    "anomaly_detection": 0.13,
    "multi_dimensional_risk": [0.7, 0.2, 0.3, 0.1]
  },
  "timestamp": "2025-09-15T11:00:37.123Z",
  "text_length": 34
}
```

---

### 2. /api/meta-cognition

- **功能**：高阶语义与异常分析，返回语义分数、异常分布等。
- **请求方式**：POST
- **参数**：`text`（string）

```
curl -X POST http://localhost:5000/api/meta-cognition -H "Content-Type: application/json" -d '{"text": "..." }'
```
- **返回**：见 /api/analyze 结构相似，增加语义和异常相关字段。

---

### 3. /api/intelligent-reasoning

- **功能**：批量/结构化推理任务。支持超长内容自动分片分析。
- **请求方式**：POST
- **参数**：
  - `text` 或 `scenario`：内容（string，支持超长文本）

```
curl -X POST http://localhost:5000/api/intelligent-reasoning -H "Content-Type: application/json" -d '{"text": "目标1内容\n目标2内容..."}'
```
- **返回**：分片分析结果、聚合主结论、各片段置信度等。

---

### 4. /api/decision

- **功能**：自动化渗透测试决策，支持单条/多条/超长文本智能分片。
- **请求方式**：POST
- **参数**：
  - `text`（string 或 string数组）

```
curl -X POST http://localhost:5000/api/decision -H "Content-Type: application/json" -d '{"text": ["http://test.com?id=1", "select * from users"]}'
```
- **返回**：每条决策详情与聚合主决策。

---

### 5. /api/exploit-chain

- **功能**：漏洞利用链批量生成与���析。支持 targets 批量目标，自动分片。
- **请求方式**：POST
- **参数**：
  - `targets`: 目标url数组或字符串
  - `operation`: 操作类型（可选，默认exploit_chain）

```
curl -X POST http://localhost:5000/api/exploit-chain -H "Content-Type: application/json" -d '{"targets": ["http://a.com", "http://b.com"]}'
```
- **返回**：每个目标的利用链分析、风险分数与聚合统计。

---

### 6. /api/security-enhancement

- **功能**：智能安全增强与威胁检测，返回威胁类型、置信度、建议等。
- **请求方式**：POST
- **参数**：`text`（string）

---

### 7. /api/creative-attacks

- **功能**：批量生成多类型创意 payload，适合自动化 fuzz。
- **请求方式**：POST
- **参数**：
  - `base_payload`：基础payload文本
  - `attack_types`：攻击类型数组（如 sql_injection, xss, rce）
  - `count`：生成数量

---

### 8. /api/vuln/verify

- **功能**：自动化漏洞验证，支持 GET/POST/自定义头/截图等。
- **参数**：
  - `target_url`/`url`/`target`
  - `payload`
  - `method` (GET/POST)
  - 其它可选参数（如 headers, cookies, take_screenshot）

---

### 9. /api/autotest

- **功能**：自动化批量渗透测试引擎，支持多目标、多类型 payload 批量 fuzz。
- **参数**：
  - `targets`/`target_url`/`url`/`target`
  - 其它可选参数

---

## 其他常用端点

- `/api/health` 系统健康检查
- `/api/status` 运行状态
- `/api/performance` 性能监控
- `/api/system/repair` 触发自愈

---

## 高级用法

- 支持超长文本/大批量目标自动分片分析，返回聚合结论
- 批量 fuzz 支持多类型攻击自动切换
- 全端点返回详细错误码与提示

---

## 安全与合规声明

本项目仅限授权安全测试和研究用途，严禁用于未授权攻击。滥用本项目造成的一切后果由使用者自行承担。

---

## 贡献与交流

欢迎提交 issue/PR 反馈问题与建议。  
如需深度定制或技术交流请联系项目维护者。

---