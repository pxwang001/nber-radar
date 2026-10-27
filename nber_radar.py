"""Core logic for NBER Radar.

Fetches NBER working paper metadata from the public TSV metadata repository
(https://data.nber.org/nber_paper_chapter_metadata/tsv/), scores papers by
keyword relevance, and optionally generates Chinese summaries via DeepSeek API.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import functools
import html
import io
import json
import os
import pickle
import re
import smtplib
import sqlite3
import shutil
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent
METADATA_BASE = "https://data.nber.org/nber_paper_chapter_metadata/tsv"
METADATA_WEEKS = 12
USER_AGENT = "NBER Radar/0.3 (+personal research alert tool)"
DEFAULT_DB_PATH = BASE_DIR / "nber_radar.sqlite"


class NberDataError(RuntimeError):
    """NBER 数据读取失败时抛出的可读错误。"""

PROGRAM_CODES = {
    "AP": "Asset Pricing",
    "CF": "Corporate Finance",
    "CH": "Children and Families",
    "DAE": "Development of the American Economy",
    "DEV": "Development Economics",
    "ED": "Economics of Education",
    "EEE": "Environment and Energy Economics",
    "EFG": "Economic Fluctuations and Growth",
    "EH": "Economic History",
    "HC": "Health Care",
    "HE": "Health Economics",
    "IFM": "International Finance and Macroeconomics",
    "IO": "Industrial Organization",
    "ITI": "International Trade and Investment",
    "LE": "Law and Economics",
    "LS": "Labor Studies",
    "ME": "Monetary Economics",
    "PE": "Public Economics",
    "POL": "Political Economy",
    "PR": "Productivity, Innovation, and Entrepreneurship",
    "URB": "Urban Economics",
}

PRIMARY_PUBLIC_TOPICS = {
    "公共财政", "公共管理", "教育政策", "健康政策", "劳动就业",
    "城市治理", "分配公平", "环境规制", "政治经济",
}
SUPPORT_PUBLIC_TOPICS = {"政策评估", "中国研究"}
PUBLIC_PROGRAM_WEIGHTS = {
    "Public Economics": 28,
    "Political Economy": 20,
    "Law and Economics": 14,
    "Economics of Education": 16,
    "Health Economics": 16,
    "Health Care": 12,
    "Labor Studies": 12,
    "Urban Economics": 14,
    "Environment and Energy Economics": 10,
    "Children and Families": 10,
    "Development Economics": 6,
}


BROAD_SEARCH_TERMS = {
    "policy", "policies", "reform", "public policy", "technology",
    "data", "government", "state", "economic", "economy", "program",
}


@dataclasses.dataclass
class QueryGroup:
    concept: str
    terms: List[str]


@dataclasses.dataclass
class QueryUnderstanding:
    groups: List[QueryGroup]
    uncovered: List[str]
    source: str
    error: str = ""


@dataclasses.dataclass
class Paper:
    nber_id: str
    title: str
    authors: str = ""
    abstract: str = ""
    url: str = ""
    published: str = ""
    programs: str = ""
    source: str = "nber"
    score: int = 0
    level: str = "C"
    reason: str = ""
    highlights: Tuple[str, ...] = ("研究领域", "核心议题", "待读摘要")
    _matched_kws: List[str] = dataclasses.field(default_factory=list)
    matched_topics: Tuple[str, ...] = ()
    cn_summary: str = ""
    cn_title: str = ""
    cn_abstract: str = ""
    cn_highlights: Tuple[str, ...] = ()
    groups_matched: int = 0
    groups_total: int = 0


TOPIC_KEYWORDS: Dict[str, Dict[str, Any]] = {
    "公共财政": {
        "highlight": "公共财政",
        "keywords": [
            "public finance", "tax", "taxation", "income tax", "corporate tax",
            "property tax", "tax credit", "eitc", "fiscal", "budget", "deficit",
            "debt", "government spending", "public spending", "expenditure",
            "subsidy", "transfer", "welfare", "social security", "medicaid",
            "medicare", "redistribution", "public debt", "fiscal policy",
        ],
    },
    "公共管理": {
        "highlight": "公共管理",
        "keywords": [
            "public administration", "public sector", "bureaucracy", "governance",
            "state capacity", "government agency", "civil service", "officials",
            "procurement", "contracting", "compliance", "enforcement", "corruption",
            "decentralization", "local government", "municipality", "county",
            "regulation", "regulatory", "administrative",
        ],
    },
    "政策评估": {
        "highlight": "政策评估",
        "keywords": [
            "policy", "policies", "reform", "program", "evaluation", "impact",
            "effect", "effects", "causal", "causality", "difference-in-differences",
            "difference in differences", "did", "regression discontinuity", "rdd",
            "randomized", "experiment", "field experiment", "quasi-experimental",
            "instrumental variable", "event study", "natural experiment",
        ],
    },
    "教育政策": {
        "highlight": "教育政策",
        "keywords": [
            "education", "school", "schools", "teacher", "teachers", "student",
            "students", "college", "university", "tuition", "scholarship",
            "voucher", "charter school", "school finance", "human capital",
        ],
    },
    "健康政策": {
        "highlight": "健康政策",
        "keywords": [
            "health", "healthcare", "health care", "hospital", "insurance",
            "medicaid", "medicare", "public health", "mental health", "mortality",
            "physician", "medical", "drug", "opioid", "pandemic",
        ],
    },
    "劳动就业": {
        "highlight": "劳动就业",
        "keywords": [
            "labor", "labour", "employment", "unemployment", "wage", "wages",
            "minimum wage", "job", "jobs", "worker", "workers", "training",
            "labor market", "earnings", "occupation", "union",
        ],
    },
    "城市治理": {
        "highlight": "城市治理",
        "keywords": [
            "urban", "city", "cities", "housing", "rent", "zoning", "land use",
            "transport", "transportation", "commute", "infrastructure", "policing",
            "crime", "neighborhood", "local labor market", "migration",
        ],
    },
    "分配公平": {
        "highlight": "分配公平",
        "keywords": [
            "inequality", "poverty", "redistribution", "mobility", "income distribution",
            "wealth", "racial", "race", "gender", "stratification", "opportunity",
            "discrimination", "intergenerational", "social mobility",
        ],
    },
    "环境规制": {
        "highlight": "环境规制",
        "keywords": [
            "environment", "energy", "climate", "carbon", "pollution", "emissions",
            "clean air", "renewable", "electricity", "fuel", "regulation", "epa",
        ],
    },
    "政治经济": {
        "highlight": "政治经济",
        "keywords": [
            "political economy", "election", "voting", "voter", "democracy",
            "campaign", "legislature", "politician", "party", "political",
            "media", "polarization", "lobbying", "public opinion",
        ],
    },
    "中国研究": {
        "highlight": "中国研究",
        "keywords": [
            "china", "chinese", "hukou", "prefecture", "cadre", "local officials",
            "state-owned", "soe", "county", "province", "provincial",
        ],
    },
    "金融市场": {
        "highlight": "金融市场",
        "keywords": [
            "asset pricing", "stock", "equity", "bond", "portfolio", "risk",
            "return", "financial market", "investment", "capital market",
            "firm value", "corporate finance", "merger", "acquisition",
        ],
    },
    "宏观经济": {
        "highlight": "宏观经济",
        "keywords": [
            "macroeconomics", "gdp", "growth", "recession", "business cycle",
            "monetary policy", "inflation", "interest rate", "central bank",
            "aggregate demand", "supply shock", "fiscal multiplier",
        ],
    },
    "国际经济": {
        "highlight": "国际经济",
        "keywords": [
            "international trade", "trade", "tariff", "export", "import",
            "globalization", "exchange rate", "foreign direct investment", "fdi",
            "capital flows", "current account", "balance of payments",
        ],
    },
    "发展经济": {
        "highlight": "发展经济",
        "keywords": [
            "development", "developing country", "emerging market", "africa",
            "india", "poverty trap", "aid", "microfinance", "rural", "agriculture",
            "village", "household survey",
        ],
    },
    "产业组织": {
        "highlight": "产业组织",
        "keywords": [
            "industrial organization", "market structure", "competition", "monopoly",
            "oligopoly", "antitrust", "entry", "pricing", "platform", "network effect",
            "digital", "technology", "innovation", "research and development",
        ],
    },
}

METHOD_KEYWORDS: Dict[str, List[str]] = {
    "因果识别": ["causal", "causality", "natural experiment", "quasi-experimental", "instrumental variable", "rdd", "regression discontinuity", "difference-in-differences", "difference in differences", "event study", "randomized", "experiment"],
    "准实验": ["quasi-experimental", "natural experiment", "difference-in-differences", "difference in differences", "rdd", "event study"],
    "实证检验": ["estimate", "estimates", "empirical", "data", "evidence", "administrative data"],
}

GENERAL_FALLBACK_HIGHLIGHTS = ["政策评估", "公共议题", "因果识别", "实证检验", "制度影响"]

CHINESE_TO_ENGLISH: Dict[str, List[str]] = {
    "低空经济": ["low altitude", "drone", "unmanned aerial", "urban air mobility"],
    "财政": ["public finance", "fiscal", "fiscal policy", "government spending"],
    "税收": ["tax", "taxation", "tax policy", "income tax", "corporate tax"],
    "税": ["tax", "taxation", "tax policy"],
    "债务": ["public debt", "fiscal deficit", "government debt"],
    "赤字": ["fiscal deficit", "budget deficit", "public debt"],
    "预算": ["budget", "government budget", "fiscal policy"],
    "社保": ["social security", "medicare", "medicaid", "welfare"],
    "福利": ["welfare", "social welfare", "redistribution", "transfer"],
    "转移支付": ["transfer", "intergovernmental transfer", "fiscal transfer"],
    "地方政府债务": ["local government debt", "municipal debt", "fiscal decentralization"],
    "财政分权": ["fiscal decentralization", "intergovernmental", "local government"],
    "公共管理": ["public administration", "public sector", "governance", "bureaucracy"],
    "治理": ["governance", "public administration", "state capacity"],
    "官僚": ["bureaucracy", "bureaucratic", "civil service", "officials"],
    "政府": ["government", "public sector", "state", "local government"],
    "地方政府": ["local government", "municipality", "county government", "decentralization"],
    "官员": ["officials", "cadre", "civil service", "bureaucracy"],
    "腐败": ["corruption", "rent-seeking", "governance"],
    "寻租": ["rent-seeking", "corruption", "lobbying"],
    "政绩": ["performance", "cadre evaluation"],
    "晋升": ["promotion", "cadre", "political incentive"],
    "政策": ["policy", "policies", "reform", "public policy"],
    "评估": ["evaluation", "impact", "effect", "program evaluation"],
    "改革": ["reform", "policy reform", "institutional change"],
    "因果": ["causal", "causality", "causal inference"],
    "实证": ["empirical", "evidence", "estimate", "causal inference"],
    "准实验": ["quasi-experimental", "natural experiment", "difference-in-differences"],
    "工具变量": ["instrumental variable", "iv", "causal identification"],
    "断点回归": ["regression discontinuity", "rdd", "causal inference"],
    "双重差分": ["difference-in-differences", "difference in differences", "did"],
    "教育": ["education", "school", "schooling", "human capital", "college"],
    "人力资本": ["human capital", "education", "skill", "training"],
    "健康": ["health", "healthcare", "public health", "health insurance"],
    "医疗": ["healthcare", "medicare", "medicaid", "health insurance"],
    "劳动": ["labor", "labour", "employment", "labor market", "worker"],
    "就业": ["employment", "unemployment", "labor market", "job"],
    "工资": ["wage", "wages", "minimum wage", "earnings"],
    "最低工资": ["minimum wage", "labor policy"],
    "不平等": ["inequality", "income distribution", "wealth inequality"],
    "贫困": ["poverty", "redistribution", "social mobility", "inequality"],
    "代际": ["intergenerational", "social mobility", "intergenerational mobility"],
    "流动": ["mobility", "social mobility", "intergenerational mobility"],
    "城市": ["urban", "city", "cities", "urban economics"],
    "住房": ["housing", "rent", "zoning", "housing market"],
    "土地": ["land", "land use", "zoning", "property"],
    "基础设施": ["infrastructure", "public investment", "transportation"],
    "交通": ["transportation", "commute", "infrastructure"],
    "环境": ["environment", "environmental", "pollution", "climate"],
    "能源": ["energy", "renewable", "electricity", "fuel"],
    "污染": ["pollution", "emissions", "environmental regulation"],
    "气候": ["climate", "climate change", "carbon"],
    "政治": ["political", "politics", "political economy"],
    "选举": ["election", "voting", "voter", "political economy"],
    "民主": ["democracy", "political participation", "voting"],
    "数字": ["digital", "digital economy", "technology", "data"],
    "数字经济": ["digital economy", "technology", "platform", "digital"],
    "平台": ["platform", "digital platform", "technology"],
    "人工智能": ["artificial intelligence", "ai", "automation", "technology"],
    "技术": ["technology", "innovation", "digital", "automation"],
    "创新": ["innovation", "technology", "research and development", "rd"],
    "中国": ["china", "chinese", "china studies", "chinese economy"],
    "户籍": ["hukou", "migration", "rural-urban"],
    "农村": ["rural", "agriculture", "rural-urban", "village"],
    "贸易": ["trade", "international trade", "tariff", "globalization"],
    "全球化": ["globalization", "international trade", "global"],
    "产业": ["industry", "industrial", "manufacturing", "industrial policy"],
    "产业政策": ["industrial policy", "industrial", "government intervention"],
    "金融": ["finance", "financial", "financial market", "credit"],
    "银行": ["bank", "banking", "credit", "financial institution"],
    "企业": ["firm", "firms", "corporate", "business", "enterprise"],
    "中小企业": ["small business", "sme", "entrepreneurship", "firm"],
    "创业": ["entrepreneurship", "startup", "small business", "innovation"],
    "增长": ["growth", "economic growth", "development"],
    "发展": ["development", "economic development", "growth"],
    "经济": ["economy", "economic", "economics", "economic policy"],
    "机器学习": ["machine learning", "deep learning", "artificial intelligence"],
    "大数据": ["big data", "data", "administrative data"],
    "随机实验": ["randomized experiment", "randomized controlled trial"],
    "自然实验": ["natural experiment", "quasi-experimental", "causal inference"],
    "宏观": ["macroeconomics", "macroeconomic", "aggregate", "gdp"],
    "货币": ["monetary", "monetary policy", "central bank", "interest rate"],
    "通货膨胀": ["inflation", "monetary policy", "price level"],
    "股票": ["stock", "equity", "asset pricing", "financial market"],
    "资产": ["asset", "asset pricing", "portfolio", "investment"],
    "耐心资本": ["patient capital", "patient investors", "long-term capital"],
    "资产定价": ["asset pricing", "asset prices", "stock returns"],
    "定价": ["pricing", "price setting"],
    "数字政府": ["digital government", "e-government", "digital public services"],
    "教育投入": ["education spending", "school spending", "educational investment", "parental investment"],
    "家庭": ["household", "family", "parents"],
    "并购": ["merger", "acquisition", "corporate finance", "firm"],
    "竞争": ["competition", "antitrust", "market structure", "industrial organization"],
    "垄断": ["monopoly", "antitrust", "market power", "competition"],
    "发展中国家": ["developing country", "development economics", "emerging market"],
    "非洲": ["africa", "development economics", "developing country"],
    "印度": ["india", "development economics", "emerging market"],
    "监管": ["regulation", "regulatory", "government regulation", "compliance"],
    "养老金": ["pension", "retirement", "social security", "elderly"],
    "老龄化": ["aging", "elderly", "population aging", "demography"],
    "生育": ["fertility", "birth rate", "childbearing", "family"],
    "碳排放": ["carbon emissions", "co2 emissions", "greenhouse gas"],
    "碳交易": ["carbon trading", "emissions trading", "carbon market"],
    "碳排放交易": ["carbon emissions trading", "cap and trade", "emissions trading"],
    "农业": ["agriculture", "agricultural", "farm", "rural"],
    "数字基础设施": ["digital infrastructure", "broadband", "internet access", "digital divide"],
    "人才": ["talent", "human capital", "skilled workers", "labor force"],
    "高校": ["university", "higher education", "college", "academic institution"],
}


ENGLISH_ALIASES = {"ai": ["artificial intelligence"]}

GENERIC_CN_TERMS = {"影响", "效果", "作用", "评估", "研究", "分析", "政策", "改革", "经济", "发展", "增长", "因果", "实证"}


def refine_search_keywords(keywords: Sequence[str]) -> List[str]:
    cleaned: List[str] = []
    seen = set()
    for keyword in keywords:
        keyword = clean_text(keyword)
        key = keyword.lower()
        if keyword and key not in seen:
            seen.add(key)
            cleaned.append(keyword)
    specific = [k for k in cleaned if k.lower() not in BROAD_SEARCH_TERMS]
    return specific if specific else cleaned


def understand_query(query: str, api_key: str = "",
                     base_url: str = "https://api.deepseek.com",
                     model: str = "deepseek-chat") -> QueryUnderstanding:
    """将中文查询拆解为概念组，支持 LLM 和词典两种模式。"""
    if not query or not query.strip():
        return QueryUnderstanding(groups=[], uncovered=[], source="dict")
    query = query.strip()

    if api_key and api_key.strip():
        return _understand_query_llm(query, api_key, base_url, model)
    else:
        return _understand_query_dict(query)


def _understand_query_llm(query: str, api_key: str, base_url: str, model: str) -> QueryUnderstanding:
    """用 LLM 拆解查询。"""
    try:
        from openai import OpenAI
    except ImportError:
        return _understand_query_dict(query)

    prompt = (
        "你是经济学文献检索助手。把用户的中文查询转成 NBER 论文题目和摘要里检索用的英文关键词。\n"
        "规则：\n"
        "1. 概念只能来自查询里明确出现的内容，数量等于查询里语义独立的部分（通常 1-3 个）。严禁添加查询里没有的概念，严禁加入 NBER、policy implications、economic analysis 之类的泛词。\n"
        "2. “影响、效果、研究、分析、机制”这类词不单独成概念。\n"
        "3. 每个概念给 3-5 个英文表达，必须是经济学论文里真实常见的写法，按常见程度排序，缩写单独成项。\n"
        "4. 只输出 JSON，不要解释。\n\n"
        "示例一\n查询：人工智能对就业的影响\n"
        "{\"groups\":[{\"concept\":\"人工智能\",\"terms\":[\"artificial intelligence\",\"AI\",\"machine learning\",\"automation\"]},"
        "{\"concept\":\"就业\",\"terms\":[\"employment\",\"jobs\",\"labor market\",\"unemployment\"]}]}\n\n"
        "示例二\n查询：地方政府债务\n"
        "{\"groups\":[{\"concept\":\"地方政府债务\",\"terms\":[\"local government debt\",\"municipal debt\",\"subnational debt\",\"municipal bonds\"]}]}\n\n"
        f"查询：{query}\n"
    )

    try:
        raw = call_llm(prompt, api_key, base_url, model, max_tokens=1500, attempts=2, timeout=30)
    except LlmError as exc:
        fallback = _understand_query_dict(query)
        return dataclasses.replace(fallback, error=f"{exc}，本次改用内置词典拆词")

    try:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        data = json.loads(match.group()) if match else {}
        groups = []
        for item in data.get("groups", []):
            concept = str(item.get("concept", "")).strip()
            terms = [str(t).strip() for t in item.get("terms", []) if str(t).strip()]
            terms = refine_search_keywords(terms)
            if terms:
                groups.append(QueryGroup(concept=concept, terms=terms))
        if groups:
            return QueryUnderstanding(groups=groups[:3], uncovered=[], source="llm")
    except Exception:
        pass
    fallback = _understand_query_dict(query)
    return dataclasses.replace(fallback, error="大模型返回格式无法解析，本次改用内置词典拆词")


def _understand_query_dict(query: str) -> QueryUnderstanding:
    """用词典拆解查询。"""
    groups = []
    covered_indices = set()
    sorted_terms = sorted(CHINESE_TO_ENGLISH.keys(), key=len, reverse=True)

    for term in sorted_terms:
        if term in query:
            idx = query.find(term)
            overlap = any(mi in covered_indices for mi in range(idx, idx + len(term)))
            if not overlap:
                if term not in GENERIC_CN_TERMS:
                    terms = refine_search_keywords(CHINESE_TO_ENGLISH[term])
                    if terms:
                        groups.append(QueryGroup(concept=term, terms=terms))
                for mi in range(idx, idx + len(term)):
                    covered_indices.add(mi)

    english_words = re.findall(r'(?<![A-Za-z])[A-Za-z]{2,}(?![A-Za-z])', query)
    english_singles = re.findall(r'(?<![A-Za-z])[A-Z](?![A-Za-z])', query)
    for word in dict.fromkeys(english_words + english_singles):
        extra = ENGLISH_ALIASES.get(word.lower(), [])
        groups.append(QueryGroup(concept=word, terms=[word.lower(), *extra]))

    uncovered = _extract_uncovered(query, groups)
    return QueryUnderstanding(groups=groups, uncovered=uncovered, source="dict")


def _extract_uncovered(query: str, groups: List[QueryGroup]) -> List[str]:
    """提取未被识别的中文片段（≥2 个汉字）。"""
    covered_indices = set()
    sorted_terms = sorted(CHINESE_TO_ENGLISH.keys(), key=len, reverse=True)
    for term in sorted_terms:
        idx = query.find(term)
        if idx >= 0:
            overlap = any(mi in covered_indices for mi in range(idx, idx + len(term)))
            if not overlap:
                for mi in range(idx, idx + len(term)):
                    covered_indices.add(mi)

    for filler in sorted(GENERIC_CN_TERMS | {"变化", "情况", "问题", "方面", "关系", "作用于"}, key=len, reverse=True):
        start = 0
        while True:
            idx = query.find(filler, start)
            if idx < 0:
                break
            covered_indices.update(range(idx, idx + len(filler)))
            start = idx + len(filler)

    stopwords = {"的", "对", "与", "和", "及", "在", "中", "于", "之"}
    for i, char in enumerate(query):
        if char in stopwords:
            covered_indices.add(i)

    english_matches = list(re.finditer(r'\b[a-zA-Z]+\b', query))
    for match in english_matches:
        for i in range(match.start(), match.end()):
            covered_indices.add(i)

    uncovered_texts = []
    i = 0
    while i < len(query):
        if i not in covered_indices and re.match(r'[一-鿿]', query[i]):
            start = i
            while i < len(query) and i not in covered_indices and re.match(r'[一-鿿]', query[i]):
                i += 1
            text = query[start:i]
            if len(text) >= 2:
                uncovered_texts.append(text)
        else:
            i += 1
    return uncovered_texts


def decompose_chinese_query(query: str) -> List[str]:
    """词典拆词，返回扁平列表，保持向后兼容。"""
    understanding = _understand_query_dict(query)
    keywords: List[str] = []
    for group in understanding.groups:
        keywords.extend(group.terms)
    for text in understanding.uncovered:
        for group in understanding.groups:
            if text.lower() in [t.lower() for t in group.terms]:
                break
        else:
            pass
    return refine_search_keywords(keywords)


def decompose_with_llm(query: str, api_key: str,
                       base_url: str = "https://api.deepseek.com",
                       model: str = "deepseek-chat") -> List[str]:
    """用 LLM 将中文查询翻译为英文搜索关键词，失败时回退到词典拆词。"""
    try:
        from openai import OpenAI
    except ImportError:
        return decompose_chinese_query(query)

    prompt = (
        "你是一名经济学研究助手。用户输入了一个中文查询，"
        "请生成适合在 NBER 经济学论文标题和摘要中搜索的英文关键词。\n"
        "要求：\n"
        "1. 输出 3-8 个关键词，每行一个\n"
        "2. 优先输出高价值的多词短语（如 'artificial intelligence'）\n"
        "3. 也可以输出关键的短单词/缩写（如 'AI'），但不要输出过于宽泛的经济学通用词\n"
        "4. 避免输出包含 'labor'、'market'、'policy' 等宽泛词的短语，除非它们与查询主题紧密相关\n"
        "5. 只输出英文，不要输出中文，不要序号，不要额外说明\n\n"
        f"用户查询：{query}"
    )

    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=30)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000,
        )
        raw = response.choices[0].message.content.strip()
        keywords = [line.strip() for line in raw.split("\n") if line.strip()]
        keywords = [re.sub(r"^[\d\.\-\*\s]+", "", kw).strip() for kw in keywords]
        keywords = [kw for kw in keywords if len(kw) > 1]

        # 从短语中提取缩写（如从 "AI Regulation" 中提取 "AI"），
        # 但不过度拆分——只保留 2-3 位全大写缩写词
        extra = []
        for kw in keywords:
            parts = kw.split()
            if len(parts) > 1:
                extra.extend(p for p in parts if re.match(r'^[A-Z]{2,3}$', p))
        keywords.extend(extra)

        seen = set()
        deduped = []
        for kw in keywords:
            kl = kw.lower()
            if kl not in seen:
                seen.add(kl)
                deduped.append(kw)
        return refine_search_keywords(deduped) if deduped else decompose_chinese_query(query)
    except Exception:
        return decompose_chinese_query(query)


def score_one_paper_custom(paper: Paper, custom_keywords: List[str], weight: float = 6.0) -> Paper:
    """把“查询词相关度”和“专业相关度”合并为一个更直观的分数。"""
    base = score_one_paper(paper)
    title_hits = keyword_hits(base.title, custom_keywords)
    abstract_hits = keyword_hits(base.abstract, custom_keywords)
    program_hits = keyword_hits(base.programs, custom_keywords)

    matched_kws = list(dict.fromkeys(title_hits + abstract_hits + program_hits))
    if not matched_kws:
        return dataclasses.replace(base, score=0, level="D", _matched_kws=[])

    # 标题命中最可靠，摘要命中次之。weight 只做轻微调节，避免用户把分数拉爆。
    scale = max(0.6, min(1.6, float(weight) / 6.0))
    title_score = len(title_hits) * 30
    abstract_only = [kw for kw in abstract_hits if kw not in title_hits]
    abstract_score = len(abstract_only) * 12
    program_only = [kw for kw in program_hits if kw not in title_hits and kw not in abstract_hits]
    program_score = len(program_only) * 5
    query_score = min(72, int(round((title_score + abstract_score + program_score) * scale)))

    # 专业相关度只占一部分：查询很准的论文不会被埋没，公共经济/管理论文会排得更靠前。
    combined = min(100, int(round(query_score + base.score * 0.35)))
    level = "A" if combined >= 70 else "B" if combined >= 50 else "C" if combined >= 30 else "D"
    where = "标题" if title_hits else "摘要"
    reason = f"{where}匹配“{'、'.join(matched_kws[:3])}”"
    if base.matched_topics:
        reason += f"，并与{base.matched_topics[0]}相关。"
    else:
        reason += "。"

    highlights = list(base.highlights)
    query_label = "标题命中" if title_hits else "摘要命中"
    highlights = [query_label] + [h for h in highlights if h != query_label]
    return dataclasses.replace(
        base,
        score=combined,
        level=level,
        reason=reason,
        highlights=tuple(highlights[:3]),
        _matched_kws=matched_kws,
    )


DEFAULT_LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
DEFAULT_LLM_MODEL = "glm-4-flash-250414"
FALLBACK_LLM_MODEL = "glm-4.7-flash"


class LlmError(RuntimeError):
    """大模型调用失败，信息可直接展示给用户。"""


def _friendly_llm_error(exc: Exception) -> str:
    text = str(exc)
    if "1305" in text or "访问量过大" in text:
        return "模型当前访问量过大（免费模型高峰期常见），请稍后再试"
    if "1302" in text or "速率限制" in text:
        return "请求过于频繁，免费模型只允许 1 个并发，请稍后再试"
    if "401" in text or "Unauthorized" in text or "令牌" in text:
        return "API Key 无效或已过期"
    if "timed out" in text.lower() or "timeout" in text.lower():
        return "模型响应超时"
    return f"大模型调用失败：{text[:120]}"


def call_llm(prompt: str, api_key: str, base_url: str = DEFAULT_LLM_BASE_URL,
             model: str = DEFAULT_LLM_MODEL, max_tokens: int = 3000,
             attempts: int = 3, timeout: int = 60) -> str:
    """调用大模型，遇到限流自动退避重试；主模型持续失败时改用备选模型。失败抛 LlmError。"""
    import time
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LlmError("未安装 openai 库") from exc

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)
    models = [model] + ([FALLBACK_LLM_MODEL] if model != FALLBACK_LLM_MODEL and "bigmodel.cn" in base_url else [])
    last: Optional[Exception] = None
    for m in models:
        for i in range(attempts):
            try:
                resp = client.chat.completions.create(
                    model=m,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
                content = (resp.choices[0].message.content or "").strip()
                if content:
                    return content
                last = RuntimeError("模型返回为空")
            except Exception as exc:
                last = exc
                text = str(exc)
                if "401" in text or "Unauthorized" in text:
                    raise LlmError(_friendly_llm_error(exc)) from exc
                if i < attempts - 1:
                    time.sleep(2 * (i + 1))
    raise LlmError(_friendly_llm_error(last or RuntimeError("未知错误")))


@dataclasses.dataclass
class Translation:
    nber_id: str
    title_zh: str = ""
    abstract_zh: str = ""
    one_liner: str = ""
    model: str = ""


def parse_translation(raw: str) -> Dict[str, str]:
    """解析【标题】【摘要】【一句话】三段式输出。"""
    out = {"title_zh": "", "abstract_zh": "", "one_liner": ""}
    pat = r"【(标题|摘要|一句话)】\s*(.*?)(?=【(?:标题|摘要|一句话)】|\Z)"
    names = {"标题": "title_zh", "摘要": "abstract_zh", "一句话": "one_liner"}
    for label, body in re.findall(pat, raw, flags=re.S):
        out[names[label]] = body.strip().strip("`").strip()
    return out


def build_glossary(text: str, extra: Optional[Dict[str, str]] = None) -> str:
    """从词典和本次检索概念里挑出文中出现的术语，作为翻译时的统一译法。"""
    lower = text.lower()
    pairs: Dict[str, str] = dict(extra or {})
    for cn, ens in CHINESE_TO_ENGLISH.items():
        if len(cn) < 2 or cn in GENERIC_CN_TERMS or not ens:
            continue
        en = ens[0].lower()
        if en not in pairs and re.search(_keyword_regex_cached(en), lower):
            pairs[en] = cn
    return "；".join(f"{en}={cn}" for en, cn in list(pairs.items())[:15])


def translate_paper(paper: Paper, api_key: str, base_url: str = DEFAULT_LLM_BASE_URL,
                    model: str = DEFAULT_LLM_MODEL,
                    glossary: Optional[Dict[str, str]] = None) -> Translation:
    """一次调用得到中文题目、完整摘要译文和一句话凝练。"""
    if not paper.title:
        raise LlmError("论文缺少标题")
    has_abs = bool(paper.abstract)
    gloss = build_glossary(f"{paper.title} {paper.abstract}", glossary)
    gloss_line = f"术语表（必须按此译法，其余术语按经济学惯例，不要按字面或日常含义翻译）：{gloss}\n" if gloss else \
        "注意：术语按经济学语境翻译，不要按日常含义直译（如 patient capital 是耐心资本，不是患者资本）。\n"
    prompt = (
        "你是经济学文献翻译助手，读者是中文社科研究者。请翻译下面的 NBER 工作论文，严格按格式输出，不要输出其他内容。\n"
        "要求：\n"
        "1. 标题：译成规范的中文学术标题，保留专有名词、模型名和缩写原文（如 DID、EITC、GPT）。\n"
        "2. 摘要：完整忠实翻译全部内容，不删减、不增补、不评论。\n"
        "3. 一句话：不超过 60 字，概括研究问题和摘要中明确写出的主要发现。摘要没有说明的方法、结论或政策含义一律不写，不要推测。\n"
        f"{gloss_line}"
        "格式：\n【标题】中文标题\n【摘要】中文摘要\n【一句话】一句话\n\n"
        f"Title: {paper.title}\n\nAbstract: {paper.abstract if has_abs else '(无摘要)'}"
    )
    raw = call_llm(prompt, api_key, base_url, model, max_tokens=3500)
    data = parse_translation(raw)
    if not data["title_zh"]:
        raise LlmError("模型输出格式不符，未能解析")
    if not has_abs:
        data["abstract_zh"] = ""
    return Translation(nber_id=paper.nber_id, model=model, **data)


def init_translation_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """CREATE TABLE IF NOT EXISTS translations (
            nber_id TEXT PRIMARY KEY, title_zh TEXT, abstract_zh TEXT,
            one_liner TEXT, model TEXT, created_at TEXT)"""
    )
    conn.commit()
    conn.close()


def save_translation(tr: Translation, db_path: str | Path = DEFAULT_DB_PATH) -> None:
    init_translation_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT OR REPLACE INTO translations VALUES (?, ?, ?, ?, ?, ?)",
        (tr.nber_id, tr.title_zh, tr.abstract_zh, tr.one_liner, tr.model,
         dt.datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def load_translations(nber_ids: Sequence[str], db_path: str | Path = DEFAULT_DB_PATH) -> Dict[str, Translation]:
    ids = [i for i in dict.fromkeys(nber_ids) if i]
    if not ids:
        return {}
    init_translation_db(db_path)
    conn = sqlite3.connect(str(db_path))
    marks = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT nber_id, title_zh, abstract_zh, one_liner, model FROM translations WHERE nber_id IN ({marks})", ids
    ).fetchall()
    conn.close()
    return {r[0]: Translation(*r) for r in rows}


def apply_translations(papers: Sequence[Paper], db_path: str | Path = DEFAULT_DB_PATH) -> List[Paper]:
    """把已缓存的中文内容写回论文对象（cn_title、cn_abstract、cn_summary）。"""
    cached = load_translations([p.nber_id for p in papers], db_path)
    out = []
    for p in papers:
        tr = cached.get(p.nber_id)
        out.append(dataclasses.replace(
            p, cn_title=tr.title_zh, cn_abstract=tr.abstract_zh, cn_summary=tr.one_liner,
        ) if tr else p)
    return out


def generate_chinese_summary(paper: Paper, api_key: str,
                             base_url: str = "https://api.deepseek.com",
                             model: str = "deepseek-chat") -> Tuple[str, List[str]]:
    """调用 LLM 生成一句话中文总结和 3 条 highlights。"""
    try:
        from openai import OpenAI
    except ImportError:
        return "", []

    if not paper.abstract and not paper.title:
        return "", []

    content = f"Title: {paper.title}"
    if paper.abstract:
        content += f"\n\nAbstract: {paper.abstract[:1500]}"

    prompt = (
        "你是一名经济学文献助手，帮助中文读者快速了解 NBER 工作论文。\n"
        "请根据以下英文论文标题和摘要，用中文输出：\n"
        "1. 一句话总结（≤50字）：概括论文的核心研究问题和主要发现\n"
        "2. 三条 highlights（每条≤20字）：分别提炼研究方法、核心发现、政策意义\n\n"
        "严格按以下 JSON 格式返回，不要包含其他内容：\n"
        '{"summary": "...", "highlights": ["...", "...", "..."]}\n\n'
        f"{content}"
    )

    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=30)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000,
        )
        raw = response.choices[0].message.content.strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            summary = str(data.get("summary", ""))
            highlights = data.get("highlights", [])
            if isinstance(highlights, list):
                highlights = [str(h) for h in highlights[:3]]
            return summary, highlights
    except Exception:
        pass
    return "", []


CACHE_DIR = Path(__file__).parent / ".nber_cache"
CACHE_FILE = CACHE_DIR / "papers.json"
API_CONFIG_PATH = CACHE_DIR / "api_config.json"
INDEX_CACHE_FILE = CACHE_DIR / "index.pkl"
CACHE_MAX_AGE = 3600  # 1 小时后缓存过期，重新抓取
TSV_CACHE_MAX_AGE = 86400  # TSV 文件缓存 24 小时
INDEX_PERSIST = True

_search_index: Optional[pd.DataFrame] = None
_index_signature: Optional[Dict[str, Tuple[float, int]]] = None


@functools.lru_cache(maxsize=512)
def _keyword_regex_cached(keyword: str) -> str:
    """编译正则表达式并缓存。"""
    keyword = clean_text(keyword).lower()
    escaped = re.escape(keyword).replace(r"\ ", r"\s+")
    if re.fullmatch(r"[a-z0-9][a-z0-9\- ]*", keyword):
        if len(keyword) >= 3 and keyword[-1].isalpha():
            if keyword.endswith("y") and not keyword.endswith(("ay", "ey", "oy", "uy")):
                escaped = escaped[:-1] + "(?:y|ies)"
            else:
                escaped += "(?:s|es)?"
        return rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
    return escaped


def _paper_to_dict(p: Paper) -> dict:
    return dataclasses.asdict(p)


def _dict_to_paper(d: dict) -> Paper:
    # 兼容旧缓存，并在读取时顺手修复乱码。
    data = dict(d)
    data["highlights"] = tuple(data.get("highlights", []))
    data["cn_highlights"] = tuple(data.get("cn_highlights", []))
    data["_matched_kws"] = list(data.get("_matched_kws", []))
    data["matched_topics"] = tuple(data.get("matched_topics", []))
    data["groups_matched"] = int(data.get("groups_matched", 0))
    data["groups_total"] = int(data.get("groups_total", 0))
    for key in ("nber_id", "title", "authors", "abstract", "url", "published",
                "programs", "source", "reason", "cn_summary"):
        if key in data:
            data[key] = clean_text(data.get(key, ""))
    return Paper(**data)


def save_papers_cache(papers: List[Paper]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "timestamp": dt.datetime.now().isoformat(),
        "papers": [_paper_to_dict(p) for p in papers],
    }
    CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")


def load_papers_cache(allow_stale: bool = False) -> Optional[List[Paper]]:
    """读取缓存。网络不可用时可通过 allow_stale 使用旧缓存。"""
    candidates = [CACHE_FILE, BASE_DIR / "papers.json"]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
            timestamp = data.get("timestamp")
            age = None
            if timestamp:
                age = (dt.datetime.now() - dt.datetime.fromisoformat(timestamp)).total_seconds()
            if not allow_stale and age is not None and age > CACHE_MAX_AGE:
                continue
            papers = [_dict_to_paper(d) for d in data.get("papers", [])]
            if papers:
                return papers
        except Exception:
            continue
    return None


def clear_cache(include_tsv: bool = True) -> None:
    """清理论文缓存；默认同时清理 TSV 缓存，确保真正重新抓取。"""
    _reset_index()
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
    if include_tsv:
        tsv_dir = CACHE_DIR / "tsv"
        if tsv_dir.exists():
            shutil.rmtree(tsv_dir, ignore_errors=True)


def load_api_config() -> dict:
    """只读取非敏感模型配置；API Key 不再从磁盘读取。"""
    if API_CONFIG_PATH.exists():
        try:
            data = json.loads(API_CONFIG_PATH.read_text(encoding="utf-8"))
            return {
                "api_base_url": data.get("api_base_url", ""),
                "api_model": data.get("api_model", ""),
            }
        except Exception:
            return {}
    return {}


def save_api_config(api_key: str = "", api_base_url: str = "", api_model: str = "") -> None:
    """只保存模型地址和模型名，绝不把 API Key 明文写入磁盘。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    API_CONFIG_PATH.write_text(
        json.dumps({"api_base_url": api_base_url, "api_model": api_model}, ensure_ascii=False),
        encoding="utf-8",
    )


def repair_mojibake(text: str) -> str:
    """修复常见的 UTF-8 被误当作 Latin-1 后产生的乱码。"""
    if not text:
        return ""
    markers = ("â", "Ã", "Â")
    if not any(marker in text for marker in markers):
        return text
    try:
        fixed = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    old_bad = sum(text.count(marker) for marker in markers)
    new_bad = sum(fixed.count(marker) for marker in markers)
    return fixed if new_bad < old_bad else text


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value)
    if "<" in text and ">" in text:
        text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = html.unescape(text)
    text = repair_mojibake(text)
    return re.sub(r"\s+", " ", text).strip()


def extract_nber_id(text: str) -> str:
    if not text:
        return ""
    match = re.search(r"\b([wht]\d{4,6})\b", text, flags=re.I)
    return match.group(1).lower() if match else ""


def normalize_date(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    try:
        parsed = dt.datetime(*value[:6]) if isinstance(value, tuple) else None
    except Exception:
        parsed = None
    if parsed:
        return parsed.date().isoformat()
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(text, fmt).date().isoformat()
        except Exception:
            pass
    return text


def fetch_tsv(filename: str) -> pd.DataFrame:
    url = f"{METADATA_BASE}/{filename}"
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=(3, 8))
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise NberDataError(f"无法连接 NBER 数据源：{exc}") from exc

    # 明确按 UTF-8 读取，避免英文弯引号出现 â€™ 一类乱码。
    decoded = resp.content.decode("utf-8-sig", errors="replace")
    return pd.read_csv(io.StringIO(decoded), sep="\t", dtype=str, engine="python", on_bad_lines="skip")


def fetch_tsv_cached(filename: str, force_refresh: bool = False) -> pd.DataFrame:
    """优先使用有效缓存；网络失败时退回项目目录中的本地数据。"""
    cache_path = CACHE_DIR / "tsv" / filename
    local_seed = BASE_DIR / filename

    if not force_refresh and cache_path.exists():
        age = (dt.datetime.now() - dt.datetime.fromtimestamp(cache_path.stat().st_mtime)).total_seconds()
        if age < TSV_CACHE_MAX_AGE:
            return pd.read_csv(cache_path, sep="\t", dtype=str, engine="python", on_bad_lines="skip")

    try:
        df = fetch_tsv(filename)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, sep="\t", index=False, encoding="utf-8")
        return df
    except NberDataError:
        fallback_paths = [cache_path, local_seed]
        for fallback in fallback_paths:
            if fallback.exists():
                return pd.read_csv(fallback, sep="\t", dtype=str, engine="python", on_bad_lines="skip")
        raise


def _program_table(force_refresh: bool = False) -> pd.DataFrame:
    try:
        prog = fetch_tsv_cached("prog.tsv", force_refresh=force_refresh)
    except Exception:
        return pd.DataFrame(columns=["paper", "program"])
    if not {"paper", "program"}.issubset(prog.columns):
        return pd.DataFrame(columns=["paper", "program"])

    prog = prog[prog["program"].notna()].copy()
    prog["program_name"] = prog["program"].map(lambda c: PROGRAM_CODES.get(c, c))
    grouped = prog.groupby("paper")["program_name"].apply(lambda x: ", ".join(dict.fromkeys(x))).reset_index()
    grouped.columns = ["paper", "program"]
    return grouped


def _get_tsv_signature() -> Dict[str, Tuple[float, int]]:
    """获取 TSV 文件的 mtime 和 size 作为签名。"""
    sig = {}
    for filename in ["ref.tsv", "abs.tsv", "prog.tsv"]:
        try:
            path = CACHE_DIR / "tsv" / filename
            if path.exists():
                stat = path.stat()
                sig[filename] = (stat.st_mtime, stat.st_size)
        except Exception:
            pass
    sig["pandas"] = pd.__version__
    return sig


def _reset_index() -> None:
    """清除内存和磁盘的索引缓存（测试用）。"""
    global _search_index, _index_signature
    _search_index = None
    _index_signature = None
    if INDEX_CACHE_FILE.exists():
        try:
            INDEX_CACHE_FILE.unlink()
        except Exception:
            pass


def _build_search_index() -> pd.DataFrame:
    """构建全库检索索引。"""
    try:
        ref = fetch_tsv_cached("ref.tsv")
    except Exception:
        ref = pd.DataFrame(columns=["paper", "author", "title", "issue_date"])

    if not ref.empty and {"paper", "title"}.issubset(ref.columns):
        ref = ref[ref["paper"].str.match(r"^w\d{4,6}$", na=False)].copy()
        ref["issue_date"] = pd.to_datetime(ref.get("issue_date"), errors="coerce")
        try:
            abs_df = fetch_tsv_cached("abs.tsv")
            if {"paper", "abstract"}.issubset(abs_df.columns):
                ref = ref.merge(abs_df[["paper", "abstract"]], on="paper", how="left")
        except Exception:
            pass
        ref = ref.merge(_program_table(), on="paper", how="left")
        ref["title_lower"] = ref.get("title", "").fillna("").astype(str).str.lower()
        ref["abstract_lower"] = ref.get("abstract", "").fillna("").astype(str).str.lower()
        ref["program_lower"] = ref.get("program", "").fillna("").astype(str).str.lower()
        for col in ("title_lower", "abstract_lower", "program_lower"):
            ref[col] = ref[col].astype(object)
        return ref
    return pd.DataFrame()


def _get_search_index() -> pd.DataFrame:
    """获取或构建检索索引（内存缓存 + 可选磁盘持久化）。"""
    global _search_index, _index_signature
    current_sig = _get_tsv_signature()

    if _search_index is not None and _index_signature == current_sig:
        return _search_index

    if INDEX_PERSIST and INDEX_CACHE_FILE.exists():
        try:
            cached_data = pickle.load(open(INDEX_CACHE_FILE, "rb"))
            if cached_data.get("signature") == current_sig:
                _search_index = cached_data["index"]
                _index_signature = current_sig
                return _search_index
        except Exception:
            pass

    _search_index = _build_search_index()
    _index_signature = current_sig
    if INDEX_PERSIST and _search_index is not None and not _search_index.empty:
        try:
            INDEX_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump({"signature": current_sig, "index": _search_index}, open(INDEX_CACHE_FILE, "wb"))
        except Exception:
            pass
    return _search_index


def _paper_from_row(row: pd.Series, date_column: str = "issue_date") -> Paper:
    nber_id = clean_text(row.get("paper", ""))
    raw_date = row.get(date_column)
    parsed = pd.to_datetime(raw_date, errors="coerce")
    published = parsed.strftime("%Y-%m-%d") if pd.notna(parsed) else ""
    return Paper(
        nber_id=nber_id,
        title=clean_text(row.get("title", "")),
        authors=clean_text(row.get("author", "")),
        abstract=clean_text(row.get("abstract", "")),
        url=f"https://www.nber.org/papers/{nber_id}",
        published=published,
        programs=clean_text(row.get("program", "")),
    )


def fetch_nber_papers(max_papers: int = 50, force_refresh: bool = False) -> List[Paper]:
    """获取最近论文。缓存保存完整近期数据，不再被首次显示数量锁死。"""
    if not force_refresh:
        cached = load_papers_cache()
        if cached is not None:
            return cached[:max_papers]

    stale = load_papers_cache(allow_stale=True) or []
    try:
        ref = fetch_tsv_cached("ref.tsv", force_refresh=force_refresh)
    except Exception:
        if stale:
            return stale[:max_papers]
        raise

    if not {"paper", "title", "issue_date"}.issubset(ref.columns):
        if stale:
            return stale[:max_papers]
        raise NberDataError("NBER 论文表缺少必要字段。")

    ref = ref[ref["paper"].str.match(r"^w\d{4,6}$", na=False)].copy()
    ref["issue_date"] = pd.to_datetime(ref["issue_date"], errors="coerce")
    cutoff = pd.Timestamp(dt.date.today() - dt.timedelta(weeks=METADATA_WEEKS))
    recent = ref[ref["issue_date"].notna() & (ref["issue_date"] >= cutoff)].copy()

    # 本地备份过旧时，宁可使用已有论文缓存，也不显示一张空表。
    if recent.empty and stale:
        return stale[:max_papers]

    recent = recent.sort_values("issue_date", ascending=False)
    try:
        abs_df = fetch_tsv_cached("abs.tsv", force_refresh=force_refresh)
        if {"paper", "abstract"}.issubset(abs_df.columns):
            recent = recent.merge(abs_df[["paper", "abstract"]], on="paper", how="left")
    except Exception:
        abstract_map = {p.nber_id: p.abstract for p in stale}
        recent["abstract"] = recent["paper"].map(abstract_map)

    recent = recent.merge(_program_table(force_refresh=force_refresh), on="paper", how="left")
    papers = [_paper_from_row(row) for _, row in recent.iterrows()]
    if papers:
        save_papers_cache(papers)
        return papers[:max_papers]
    if stale:
        return stale[:max_papers]
    return []


def list_programs() -> List[Tuple[str, str]]:
    """获取索引中实际存在的 NBER 分类，返回 (代码, 英文名) 列表，按代码字母序排序。"""
    index = _get_search_index()
    if index.empty:
        return []
    prog_str = index.get("program", "").fillna("").astype(str)
    programs_set = set()
    for prog_text in prog_str:
        if prog_text.strip():
            for prog in prog_text.split(","):
                prog = prog.strip()
                if prog:
                    programs_set.add(prog)
    reverse_mapping = {v: k for k, v in PROGRAM_CODES.items()}
    result = []
    for prog_name in sorted(programs_set):
        prog_code = reverse_mapping.get(prog_name, prog_name)
        result.append((prog_code, prog_name))
    result.sort(key=lambda x: x[0])
    return result


def parse_edited_groups(
    group_edits: List[str],
    new_group_text: str = "",
    max_groups: int = 5,
    max_terms_per_group: int = 12,
) -> List[QueryGroup]:
    """解析用户编辑的概念组文本，返回清理后的 QueryGroup 列表。

    Args:
        group_edits: 用户编辑的每个现有概念组的英文词列表（逗号分隔），
                     如 ["term1, term2", "term3"] 表示第一组有两个词，第二组一个词。
                     空字符串表示该组被删除。
        new_group_text: 新增概念组的英文词（逗号分隔）。
        max_groups: 最多允许的概念组数（默认 5）。
        max_terms_per_group: 每组最多允许的词数（默认 12）。

    Returns:
        清理后的 QueryGroup 列表，去空、去重、统一小写。
    """
    result: List[QueryGroup] = []

    for edit_text in group_edits:
        if not edit_text or not edit_text.strip():
            continue
        terms = [t.strip().lower() for t in edit_text.split(",")]
        terms = [t for t in terms if t]
        terms = list(dict.fromkeys(terms))
        if len(terms) > max_terms_per_group:
            terms = terms[:max_terms_per_group]
        if terms:
            result.append(QueryGroup(concept="手动调整", terms=terms))

    if new_group_text and new_group_text.strip():
        terms = [t.strip().lower() for t in new_group_text.split(",")]
        terms = [t for t in terms if t]
        terms = list(dict.fromkeys(terms))
        if len(terms) > max_terms_per_group:
            terms = terms[:max_terms_per_group]
        if terms:
            result.append(QueryGroup(concept="手动调整", terms=terms))

    if len(result) > max_groups:
        result = result[:max_groups]

    return result


def search_nber_papers(
    query_input: Union[List[str], List[QueryGroup], QueryUnderstanding],
    max_results: int = 20,
    weight: float = 6.0,
    min_score: int = 0,
    use_feedback: bool = True,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    programs: Optional[List[str]] = None,
) -> List[Paper]:
    """全库搜索：按概念组逐个匹配，然后排序。

    query_input 可以是：
    - List[str]（旧的扁平列表，视为一个"或"组）
    - List[QueryGroup]（概念组列表）
    - QueryUnderstanding（完整理解）

    year_from, year_to: 按 issue_date 年份筛选（含边界）
    programs: 按 NBER 分类筛选（代码或英文名称列表，多选为 OR）
    """
    groups: List[QueryGroup] = []

    if isinstance(query_input, QueryUnderstanding):
        groups = query_input.groups
    elif isinstance(query_input, list):
        if not query_input:
            return []
        if isinstance(query_input[0], QueryGroup):
            groups = query_input
        else:
            keywords = refine_search_keywords(query_input)
            if not keywords:
                return []
            groups = [QueryGroup(concept="query", terms=keywords)]

    index = _get_search_index()
    stale = load_papers_cache(allow_stale=True) or []
    n_groups = len(groups)
    if n_groups == 0:
        return []

    top: List[Paper] = []
    if not index.empty:
        mask = np.ones(len(index), dtype=bool)
        if year_from is not None or year_to is not None:
            years = index["issue_date"].dt.year
            if year_from is not None:
                mask &= years >= year_from
            if year_to is not None:
                mask &= years <= year_to
        if programs:
            prog_lower = index["program_lower"]
            prog_mask = np.zeros(len(index), dtype=bool)
            for prog in programs:
                prog_lower_str = prog.lower()
                prog_full_name = PROGRAM_CODES.get(prog, prog)
                prog_full_name_lower = prog_full_name.lower()
                prog_mask |= prog_lower.str.contains(prog_lower_str, regex=False, na=False).to_numpy(dtype=bool)
                prog_mask |= prog_lower.str.contains(prog_full_name_lower, regex=False, na=False).to_numpy(dtype=bool)
            mask &= prog_mask

        title_s, abs_s, prog_s = index["title_lower"], index["abstract_lower"], index["program_lower"]
        group_scores = pd.DataFrame(0, index=index.index, columns=range(n_groups), dtype="int64")
        for gi, group in enumerate(groups):
            patterns = [_keyword_regex_cached(t) for t in group.terms]
            pattern = "(?:" + "|".join(patterns) + ")"
            literals = [clean_text(t).lower() for t in group.terms]
            for series, points in ((prog_s, 5), (abs_s, 12), (title_s, 30)):
                pre = np.zeros(len(index), dtype=bool)
                for lit in literals:
                    pre |= series.str.contains(lit, regex=False, na=False).to_numpy(dtype=bool)
                hit = np.zeros(len(index), dtype=bool)
                if pre.any():
                    hit[pre] = series[pre].str.contains(pattern, regex=True, na=False).to_numpy(dtype=bool)
                col = group_scores[gi].to_numpy()
                group_scores[gi] = np.where(hit, np.maximum(col, points), col)
        matched_groups = (group_scores > 0).sum(axis=1)
        score_pct = (100 * group_scores.sum(axis=1) / (30 * n_groups)).round().astype(int)
        keep = mask & (matched_groups > 0) & (score_pct >= min_score)
        ranked = index[keep].assign(_mg=matched_groups[keep], _sc=score_pct[keep])
        ranked = ranked.sort_values("issue_date", ascending=False)
        ranked = ranked.sort_values(["_mg", "_sc"], ascending=False, kind="stable")
        for _, row in ranked.head(max_results * 3 if use_feedback else max_results).iterrows():
            top.append(dataclasses.replace(
                _paper_from_row(row),
                score=int(row["_sc"]),
                groups_matched=int(row["_mg"]),
                groups_total=n_groups,
            ))

    by_id = {p.nber_id: p for p in top}
    all_keywords = [t for g in groups for t in g.terms]
    for p in stale:
        if p.nber_id not in by_id and keyword_hits(f"{p.title} {p.abstract} {p.programs}", all_keywords):
            by_id[p.nber_id] = p

    scored: List[Paper] = []
    for p in by_id.values():
        if p.groups_total == 0:
            text = {"t": p.title.lower(), "a": p.abstract.lower(), "p": p.programs.lower()}
            gs = []
            for group in groups:
                best = 0
                for term in group.terms:
                    pat = _keyword_regex_cached(term)
                    if re.search(pat, text["t"]):
                        best = 30
                        break
                    if re.search(pat, text["a"]):
                        best = max(best, 12)
                    elif re.search(pat, text["p"]):
                        best = max(best, 5)
                gs.append(best)
            p = dataclasses.replace(
                p,
                score=round(100 * sum(gs) / (30 * n_groups)),
                groups_matched=sum(1 for x in gs if x > 0),
                groups_total=n_groups,
            )
        if p.groups_matched == 0 or p.score < min_score:
            continue
        scored.append(p)

    if use_feedback:
        scored = apply_feedback_preferences(scored, apply_bonus=False)
    scored.sort(key=lambda p: p.published, reverse=True)
    scored.sort(key=lambda p: (p.groups_matched, p.score), reverse=True)
    final = scored[:max_results]

    result: List[Paper] = []
    for p in final:
        level = "A" if p.score >= 70 else "B" if p.score >= 50 else "C" if p.score >= 30 else "D"
        topics = _get_matched_topics(p, all_keywords)
        group_hit = [1 if keyword_hits(f"{p.title} {p.abstract} {p.programs}", g.terms) else 0 for g in groups]
        result.append(dataclasses.replace(
            p,
            level=level,
            reason=_make_reason(p, groups, topics, group_hit),
            matched_topics=tuple(topics[:5]),
        ))
    return result


def _get_matched_topics(paper: Paper, keywords: List[str]) -> List[str]:
    """提取论文匹配的主题标签。"""
    matched = []
    full_text = f"{paper.title} {paper.abstract} {paper.programs}".lower()
    for topic, spec in TOPIC_KEYWORDS.items():
        for kw in spec["keywords"]:
            if re.search(_keyword_regex_cached(kw), full_text):
                if topic not in matched:
                    matched.append(topic)
                break
    return matched[:5]


def _make_reason(paper: Paper, groups: List[QueryGroup], matched_topics: List[str], group_scores: List[int]) -> str:
    """构造"推荐理由"。"""
    parts = []
    for group in groups:
        concept = group.concept if group.concept != "query" else ""
        if concept:
            parts.append(concept)

    reason = "命中" + "、".join(parts[:3]) if parts else ""
    if matched_topics:
        reason += f"；涉及概念：{', '.join(matched_topics[:3])}"
    groups_text = f"命中 {len([s for s in group_scores if s > 0])}/{len(groups)} 个概念"
    return reason + ("；" + groups_text if reason else groups_text)


def _keyword_regex(keyword: str) -> str:
    """为英文单词和短语加完整边界（分发给缓存版本）。"""
    return _keyword_regex_cached(keyword)


def keyword_hits(text: str, keywords: Sequence[str]) -> List[str]:
    lower = clean_text(text).lower()
    hits: List[str] = []
    for kw in keywords:
        if not kw:
            continue
        if re.search(_keyword_regex_cached(kw), lower, flags=re.I):
            hits.append(kw)
    return list(dict.fromkeys(hits))


def score_one_paper(paper: Paper) -> Paper:
    """计算论文与公共经济、公共管理的相关度。"""
    title_text = paper.title
    full_text = f"{paper.title} {paper.abstract} {paper.programs}"
    all_topic_scores: List[Tuple[str, int]] = []

    for topic, spec in TOPIC_KEYWORDS.items():
        hits = keyword_hits(full_text, spec["keywords"])
        title_hits = keyword_hits(title_text, spec["keywords"])
        if hits or title_hits:
            topic_score = min(32, len(hits) * 4 + len(title_hits) * 9)
            all_topic_scores.append((topic, topic_score))

    all_topic_scores.sort(key=lambda item: item[1], reverse=True)
    primary_scores = [(t, s) for t, s in all_topic_scores if t in PRIMARY_PUBLIC_TOPICS]
    support_scores = [(t, s) for t, s in all_topic_scores if t in SUPPORT_PUBLIC_TOPICS]

    topic_total = sum(score for _, score in primary_scores[:3])
    has_public_anchor = bool(primary_scores)

    program_score = 0
    for program, bonus in PUBLIC_PROGRAM_WEIGHTS.items():
        if keyword_hits(paper.programs, [program]):
            program_score = max(program_score, bonus)
    has_public_anchor = has_public_anchor or program_score > 0

    # “政策评估”“中国研究”只能作为加分项，不能单独把普通经济学论文推成高相关。
    support_score = min(14, sum(score for _, score in support_scores[:2])) if has_public_anchor else 0

    method_score = 0
    method_highlights: List[str] = []
    if has_public_anchor:
        for label, kws in METHOD_KEYWORDS.items():
            hits = keyword_hits(full_text, kws)
            if hits:
                method_score += min(5, len(hits) * 2)
                method_highlights.append(label)
        method_score = min(10, method_score)

    total = min(100, topic_total + program_score + support_score + method_score)
    matched_topics = [t for t, _ in primary_scores + support_scores]

    highlights: List[str] = []
    for topic in matched_topics:
        label = TOPIC_KEYWORDS[topic]["highlight"]
        if label not in highlights:
            highlights.append(label)
    for label in method_highlights:
        if label not in highlights:
            highlights.append(label)
    for fallback in GENERAL_FALLBACK_HIGHLIGHTS:
        if fallback not in highlights:
            highlights.append(fallback)

    level = "A" if total >= 70 else "B" if total >= 50 else "C" if total >= 30 else "D"
    if matched_topics:
        reason = f"与{matched_topics[0]}相关"
        if program_score:
            reason += "，且NBER分类与本网站方向相符。"
        else:
            reason += "。"
    elif program_score:
        reason = "NBER分类与公共经济或公共管理相关。"
    else:
        reason = "与公共经济、公共管理的直接联系较弱。"

    return dataclasses.replace(
        paper,
        score=int(total),
        level=level,
        reason=reason,
        highlights=tuple(highlights[:3]),
        matched_topics=tuple(matched_topics[:5]),
    )


def score_papers(papers: Iterable[Paper], min_score: int = 0) -> List[Paper]:
    scored = [score_one_paper(p) for p in papers]
    if min_score > 0:
        scored = [p for p in scored if p.score >= min_score]
    return sorted(scored, key=lambda p: (p.score, p.published), reverse=True)


def recommend_public_papers(
    papers: Iterable[Paper],
    max_results: int = 50,
    min_score: int = 25,
    use_feedback: bool = True,
) -> List[Paper]:
    scored = [score_one_paper(p) for p in papers]
    if use_feedback:
        scored = apply_feedback_preferences(scored)
    scored = [p for p in scored if p.score >= min_score]
    scored.sort(key=lambda p: (p.score, p.published), reverse=True)
    return scored[:max_results]


def data_freshness_notice(papers: Sequence[Paper], max_age_days: int = 21) -> str:
    """根据论文发布日期判断“最新数据”是否可能已经过旧。"""
    dates = [pd.to_datetime(p.published, errors="coerce") for p in papers if p.published]
    dates = [d for d in dates if pd.notna(d)]
    if not dates:
        return "无法确认当前数据的更新时间。"
    latest = max(dates).date()
    age_days = (dt.date.today() - latest).days
    if age_days > max_age_days:
        return (
            f"当前数据中最新论文日期为 {latest.isoformat()}，距今 {age_days} 天。"
            "本次可能使用了本地备用数据，请联网后点击“立即刷新”。"
        )
    return ""


def papers_to_dataframe(papers: Sequence[Paper]) -> pd.DataFrame:
    rows = []
    for p in papers:
        rows.append({
            "level": p.level,
            "score": p.score,
            "nber_id": p.nber_id,
            "title": p.title,
            "authors": p.authors,
            "published": p.published,
            "programs": p.programs,
            "reason": p.cn_summary if p.cn_summary else p.reason,
            "highlight_1": p.cn_highlights[0] if p.cn_highlights else p.highlights[0],
            "highlight_2": p.cn_highlights[1] if len(p.cn_highlights) > 1 else (p.highlights[1] if len(p.highlights) > 1 else ""),
            "highlight_3": p.cn_highlights[2] if len(p.cn_highlights) > 2 else (p.highlights[2] if len(p.highlights) > 2 else ""),
            "topics": ", ".join(p.matched_topics),
            "url": p.url,
        })
    return pd.DataFrame(rows)


def render_markdown_digest(papers: Sequence[Paper], title: str = "NBER Radar") -> str:
    today = dt.date.today().isoformat()
    lines = [f"# {title}", "", f"生成日期：{today}", "", f"筛选结果：{len(papers)} 篇", ""]
    for i, p in enumerate(papers, 1):
        summary = p.cn_summary if p.cn_summary else p.reason
        hl = list(p.cn_highlights) if p.cn_highlights else list(p.highlights)
        while len(hl) < 3:
            hl.append("—")
        lines.extend([
            f"## {i}. {p.title}", "",
            f"- 等级：{p.level}",
            f"- 相关性：{p.score}/100",
            f"- 作者：{p.authors or 'N/A'}",
            f"- NBER编号：{p.nber_id or 'N/A'}",
            f"- 日期：{p.published or 'N/A'}",
            f"- 方向：{', '.join(p.matched_topics) or p.programs or '待复核'}",
            f"- 一句话：{summary}",
            "- Highlights：",
            f"  - {hl[0]}",
            f"  - {hl[1]}",
            f"  - {hl[2]}",
            f"- 链接：{p.url}",
            "",
            "---",
            "",
        ])
    return "\n".join(lines)


def papers_to_ris(papers: Sequence[Paper]) -> str:
    """导出为 RIS 格式（EndNote/Zotero 通用）。"""
    lines = []
    for paper in papers:
        lines.append("TY  - RPRT")
        lines.append(f"TI  - {paper.title}")

        if paper.authors:
            if ";" in paper.authors:
                authors = [a.strip() for a in paper.authors.split(";")]
            else:
                authors = [a.strip() for a in paper.authors.split(",")]
            for author in authors:
                if author:
                    lines.append(f"AU  - {author}")

        if paper.published:
            lines.append(f"PY  - {paper.published[:4]}")
            lines.append(f"DA  - {paper.published}")

        lines.append("T3  - NBER Working Paper Series")

        if paper.nber_id:
            num = paper.nber_id.replace("w", "") if paper.nber_id.startswith("w") else paper.nber_id
            lines.append(f"VL  - {num}")

        if paper.url:
            lines.append(f"UR  - {paper.url}")

        if paper.abstract:
            lines.append(f"AB  - {paper.abstract}")

        lines.append("M3  - NBER Working Paper")

        note_parts = []
        if paper.programs:
            note_parts.append(f"Programs: {paper.programs}")
        if paper.cn_title:
            note_parts.append(f"中文题目: {paper.cn_title}")
        if note_parts:
            lines.append(f"N1  - {'; '.join(note_parts)}")

        lines.append("ER  - ")
        lines.append("")

    return "\n".join(lines)


def papers_to_bibtex(papers: Sequence[Paper]) -> str:
    """导出为 BibTeX 格式，特殊字符转义。"""
    def escape_bibtex(text: str) -> str:
        if not text:
            return text
        BACKSLASH_PLACEHOLDER = "\x00BACKSLASH\x00"
        text = text.replace("\\", BACKSLASH_PLACEHOLDER)
        replacements = [
            ("&", r"\&"),
            ("%", r"\%"),
            ("$", r"\$"),
            ("#", r"\#"),
            ("_", r"\_"),
            ("{", r"\{"),
            ("}", r"\}"),
            ("~", r"\textasciitilde{}"),
            ("^", r"\textasciicircum{}"),
        ]
        for old, new in replacements:
            text = text.replace(old, new)
        text = text.replace(BACKSLASH_PLACEHOLDER, r"\textbackslash{}")
        return text

    def format_title(title: str) -> str:
        if not title:
            return "{}"
        escaped = escape_bibtex(title)
        return "{" + escaped + "}"

    lines = []
    for paper in papers:
        key = f"nber{paper.nber_id.replace('w', '') if paper.nber_id else 'unknown'}"
        lines.append(f"@techreport{{{key},")

        if paper.title:
            lines.append(f"  title = {format_title(paper.title)},")

        if paper.authors:
            if ";" in paper.authors:
                authors = [a.strip() for a in paper.authors.split(";")]
            else:
                authors = [a.strip() for a in paper.authors.split(",")]
            authors = [escape_bibtex(a) for a in authors if a]
            author_str = " and ".join(authors)
            lines.append(f"  author = {{{author_str}}},")

        lines.append("  institution = {National Bureau of Economic Research},")
        lines.append("  type = {Working Paper},")

        if paper.nber_id:
            num = paper.nber_id.replace("w", "") if paper.nber_id.startswith("w") else paper.nber_id
            lines.append(f"  number = {{{num}}},")

        if paper.published:
            year = paper.published[:4]
            lines.append(f"  year = {{{year}}},")
            if len(paper.published) >= 7:
                month = paper.published[5:7]
                month_map = {
                    "01": "January", "02": "February", "03": "March", "04": "April",
                    "05": "May", "06": "June", "07": "July", "08": "August",
                    "09": "September", "10": "October", "11": "November", "12": "December",
                }
                if month in month_map:
                    lines.append(f"  month = {{{month_map[month]}}},")

        if paper.url:
            lines.append(f"  url = {{{escape_bibtex(paper.url)}}},")

        if paper.abstract:
            escaped_abstract = escape_bibtex(paper.abstract)
            lines.append(f"  abstract = {{{escaped_abstract}}},")

        lines[-1] = lines[-1].rstrip(",")
        lines.append("}")
        lines.append("")

    return "\n".join(lines)


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nber_id TEXT,
            title TEXT,
            url TEXT,
            action TEXT,
            note TEXT,
            created_at TEXT
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_feedback_nber_id ON feedback(nber_id)")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS saved_topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            groups_json TEXT NOT NULL,
            year_from INTEGER,
            year_to INTEGER,
            programs_json TEXT,
            created_at TEXT,
            last_pushed_at TEXT
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_saved_topics_name ON saved_topics(name)")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS pushed_papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic_id INTEGER,
            nber_id TEXT,
            pushed_at TEXT,
            FOREIGN KEY (topic_id) REFERENCES saved_topics(id) ON DELETE CASCADE
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pushed_papers_topic ON pushed_papers(topic_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pushed_papers_nber_id ON pushed_papers(nber_id)")
    conn.commit()
    conn.close()


def save_feedback(
    paper: Paper,
    action: str,
    note: str = "",
    db_path: str | Path = DEFAULT_DB_PATH,
) -> None:
    """同一论文只保留最新选择，避免收藏和忽略互相冲突。"""
    if action not in {"saved", "ignored"}:
        raise ValueError("action 只能是 saved 或 ignored")
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("DELETE FROM feedback WHERE nber_id = ?", (paper.nber_id,))
    cur.execute(
        "INSERT INTO feedback (nber_id, title, url, action, note, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (paper.nber_id, paper.title, paper.url, action, note, dt.datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()


def list_feedback(db_path: str | Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query("SELECT * FROM feedback ORDER BY created_at DESC", conn)
    conn.close()
    return df


def apply_feedback_preferences(
    papers: Sequence[Paper],
    db_path: str | Path = DEFAULT_DB_PATH,
    apply_bonus: bool = True,
) -> List[Paper]:
    """忽略的论文不再出现；apply_bonus=True 时，收藏方向对相似论文做轻微加分。"""
    try:
        feedback = list_feedback(db_path)
    except Exception:
        return list(papers)
    if feedback.empty:
        return list(papers)

    ignored_ids = set(feedback.loc[feedback["action"] == "ignored", "nber_id"].dropna())
    result: List[Paper] = []
    for paper in papers:
        if paper.nber_id in ignored_ids:
            continue
        if apply_bonus:
            saved_rows = feedback[feedback["action"] == "saved"]
            saved_ids = set(saved_rows["nber_id"].dropna())
            preferred_topics: Dict[str, int] = {}
            for _, row in saved_rows.iterrows():
                seed = score_one_paper(Paper(nber_id=str(row.get("nber_id", "")), title=clean_text(row.get("title", ""))))
                for topic in seed.matched_topics:
                    preferred_topics[topic] = preferred_topics.get(topic, 0) + 1
            bonus = 8 if paper.nber_id in saved_ids else 0
            bonus += min(10, sum(2 * preferred_topics.get(t, 0) for t in paper.matched_topics))
            if bonus:
                new_score = min(100, paper.score + bonus)
                new_level = "A" if new_score >= 70 else "B" if new_score >= 50 else "C" if new_score >= 30 else "D"
                paper = dataclasses.replace(paper, score=new_score, level=new_level)
        result.append(paper)
    return result


def save_topic(
    name: str,
    groups: List[QueryGroup],
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    programs: Optional[List[str]] = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """保存检索主题，返回 topic_id。同名主题会被覆盖，复用旧 topic_id。"""
    if not name or not name.strip():
        raise ValueError("主题名称不能为空")
    if not groups:
        raise ValueError("主题需要至少一个搜索分组")
    name = name.strip()
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    cur = conn.cursor()
    groups_json = json.dumps([{"concept": g.concept, "terms": g.terms} for g in groups])
    programs_json = json.dumps(programs or [])
    now = dt.datetime.now().isoformat(timespec="seconds")
    cur.execute(
        "SELECT id FROM saved_topics WHERE name = ?",
        (name,),
    )
    existing = cur.fetchone()
    if existing:
        topic_id = existing[0]
        cur.execute(
            "UPDATE saved_topics SET groups_json = ?, year_from = ?, year_to = ?, programs_json = ? WHERE id = ?",
            (groups_json, year_from, year_to, programs_json, topic_id),
        )
    else:
        cur.execute(
            "INSERT INTO saved_topics (name, groups_json, year_from, year_to, programs_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (name, groups_json, year_from, year_to, programs_json, now),
        )
        topic_id = cur.lastrowid
    conn.commit()
    conn.close()
    return topic_id


def list_topics(db_path: str | Path = DEFAULT_DB_PATH) -> List[Tuple[int, str, str]]:
    """列出所有已保存的主题，返回 (id, name, created_at) 列表。"""
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT id, name, created_at FROM saved_topics ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return rows


def load_topic(topic_id: int, db_path: str | Path = DEFAULT_DB_PATH) -> Optional[Tuple[str, QueryUnderstanding, int, int, List[str]]]:
    """加载主题，返回 (name, QueryUnderstanding, year_from, year_to, programs)，不存在时返回 None。"""
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT name, groups_json, year_from, year_to, programs_json FROM saved_topics WHERE id = ?",
        (topic_id,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    name, groups_json, year_from, year_to, programs_json = row
    groups_data = json.loads(groups_json)
    groups = [QueryGroup(concept=g["concept"], terms=g["terms"]) for g in groups_data]
    understanding = QueryUnderstanding(groups=groups, uncovered=[], source="saved_topic")
    programs = json.loads(programs_json) if programs_json else []
    return (name, understanding, year_from, year_to, programs)


def delete_topic(topic_id: int, db_path: str | Path = DEFAULT_DB_PATH) -> bool:
    """删除主题及其关联的推送记录，成功返回 True。外键约束启用，cascading delete 自动删除 pushed_papers。"""
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    cur = conn.cursor()
    cur.execute("DELETE FROM saved_topics WHERE id = ?", (topic_id,))
    affected = cur.rowcount
    conn.commit()
    conn.close()
    return affected > 0


def mark_pushed(topic_id: int, nber_ids: List[str], db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """标记论文为已推送，用于去重。同时更新主题的 last_pushed_at。"""
    if not nber_ids:
        return
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    cur = conn.cursor()
    now = dt.datetime.now().isoformat(timespec="seconds")
    for nber_id in nber_ids:
        cur.execute(
            "INSERT OR IGNORE INTO pushed_papers (topic_id, nber_id, pushed_at) VALUES (?, ?, ?)",
            (topic_id, nber_id, now),
        )
    cur.execute(
        "UPDATE saved_topics SET last_pushed_at = ? WHERE id = ?",
        (now, topic_id),
    )
    conn.commit()
    conn.close()


def list_pushed_ids(topic_id: int, db_path: str | Path = DEFAULT_DB_PATH) -> set:
    """获取已推送过的论文 ID 集合。"""
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT nber_id FROM pushed_papers WHERE topic_id = ?",
        (topic_id,),
    ).fetchall()
    conn.close()
    return {row[0] for row in rows}


def send_email_digest(markdown_body: str, subject: str = "NBER Radar") -> None:
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    sender = os.getenv("EMAIL_FROM", user or "")
    recipient = os.getenv("EMAIL_TO")
    if not all([host, user, password, sender, recipient]):
        raise RuntimeError("邮件推送需要设置 SMTP_HOST, SMTP_USER, SMTP_PASSWORD, EMAIL_FROM, EMAIL_TO。")

    msg = MIMEText(markdown_body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient

    with smtplib.SMTP(host, port) as smtp:
        smtp.starttls()
        smtp.login(user, password)
        smtp.sendmail(sender, [recipient], msg.as_string())


def save_outputs(papers: Sequence[Paper], out_dir: str = "output") -> Tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    df = papers_to_dataframe(papers)
    csv_path = out / f"nber_{today}.csv"
    md_path = out / f"nber_{today}.md"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(render_markdown_digest(papers), encoding="utf-8")
    return csv_path, md_path
