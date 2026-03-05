from __future__ import annotations

from typing import Any, Dict, List, Optional

import cohere
from langchain_cohere import ChatCohere
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from tools import get_trend_search_tool
from vector_store import get_brand_voice_retriever


# -----------------------------
# Models
# -----------------------------

class Insight(BaseModel):
    """Structured, evidence-backed insight tied to a specific chart."""

    text: str
    chart_ref: int
    source: str

    def __str__(self) -> str:
        # Keeps existing Streamlit rendering simple by showing just the text
        return self.text


class Evidence(BaseModel):
    """Captured research evidence used to ground insights."""

    query: str
    source: str
    snippet: str


class MarketingState(BaseModel):
    """Shared state for the Trend-Post AI LangGraph."""

    goal: Optional[str] = None
    processed_goal: Optional[str] = None

    research_results: List[str] = Field(default_factory=list)

    insights: List[Insight] = Field(
        default_factory=list,
        description="Insights linked to charts with data sources",
    )

    generated_post: Optional[str] = None
    error: Optional[str] = None

    # Internal config
    cohere_api_key: Optional[str] = Field(default=None, exclude=True)
    tavily_api_key: Optional[str] = Field(default=None, exclude=True)

    # Structured analytics
    region_data: List[Dict[str, Any]] = Field(default_factory=list)
    age_data: List[Dict[str, Any]] = Field(default_factory=list)
    sentiment_data: Dict[str, float] = Field(default_factory=dict)

    # Visualization specs
    visual_specs: List[Dict[str, Any]] = Field(default_factory=list)

    # Evidence captured during research
    evidence: List[Evidence] = Field(default_factory=list)

    # Cross-signal trend intelligence
    trend_signals: List[Dict[str, Any]] = Field(default_factory=list)

    competitor_signals: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detected competitor mentions and activity signals.",
    )

    scenario_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Simulated outcomes for alternate campaign scenarios.",
    )

    platform_strategies: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Mapped social platform strategies derived from simulator context.",
    )


# -----------------------------
# Nodes
# -----------------------------

def input_processor(state: MarketingState) -> MarketingState:
    if not state.goal:
        state.error = "No marketing goal provided."
        return state

    state.processed_goal = state.goal.strip()
    return state


def research_agent(state: MarketingState) -> MarketingState:
    if state.error:
        return state

    try:
        search_tool = get_trend_search_tool(state.tavily_api_key)
        query = (
            f"Recent Reddit threads and discussions about {state.processed_goal}. "
            "Focus on:\n"
            "- What users are complaining about\n"
            "- What they are excited about\n"
            "- Questions they are asking\n"
            "- Popular threads with high engagement\n"
            "Return thread titles, subreddit names, links, and summaries."
        )
        results = search_tool.invoke({"query": query})

        texts: List[str] = []
        evidence_items: List[Evidence] = []

        if isinstance(results, list):
            for r in results:
                if isinstance(r, dict):
                    snippet = (
                        r.get("snippet")
                        or r.get("content")
                        or r.get("title")
                        or ""
                    )
                    source = (
                        r.get("source")
                        or r.get("url")
                        or r.get("title")
                        or "Tavily"
                    )
                    if snippet:
                        texts.append(str(snippet))
                    evidence_items.append(
                        Evidence(
                            query=query,
                            source=str(source),
                            snippet=str(snippet)[:280],
                        )
                    )
                else:
                    text = str(r)
                    texts.append(text)
                    evidence_items.append(
                        Evidence(query=query, source="Tavily", snippet=text[:280])
                    )
        else:
            text = str(results)
            texts.append(text)
            evidence_items.append(
                Evidence(query=query, source="Tavily", snippet=text[:280])
            )

        state.research_results = [t for t in texts if t]
        state.evidence = evidence_items

    except Exception as exc:
        state.error = f"Research step failed: {exc}"

    return state


def competitor_signal_agent(state: MarketingState) -> MarketingState:
    """Legacy node; competitor signals have been disabled."""
    # Preserve node in the graph but always emit an empty list
    # so that the rest of the workflow remains unchanged.
    state.competitor_signals = []
    return state


def data_extractor(state: MarketingState) -> MarketingState:
    if state.error:
        return state

    import json

    try:
        if not state.research_results:
            state.error = "No research results to extract data from."
            return state

        # Use structured JSON output to avoid brittle parsing.
        client = cohere.ClientV2(api_key=state.cohere_api_key)

        research_text = "\n\n".join(state.research_results[:8])
        prompt = (
            "You are a marketing data analyst.\n"
            "Extract or infer demographic + sentiment metrics from the research.\n"
            "If exact numbers are missing, infer realistic values from context.\n"
            "Geography must be CITIES (top 5). If a country is clearly mentioned,\n"
            "pick cities within that country; otherwise pick globally relevant cities.\n\n"
            "Return ONLY a JSON object with this exact shape:\n"
            "{\n"
            '  "region_data": [{"region": "City", "score": 0-100}],\n'
            '  "age_data": [\n'
            '    {"group": "Gen Z", "score": 0-100},\n'
            '    {"group": "Millennials", "score": 0-100},\n'
            '    {"group": "Gen X", "score": 0-100},\n'
            '    {"group": "Boomers", "score": 0-100}\n'
            "  ],\n"
            '  "sentiment_data": {"positive": 0-100, "neutral": 0-100, "negative": 0-100}\n'
            "}\n\n"
            "Rules:\n"
            "- Use numbers (not strings)\n"
            "- Keep region_data scores between 0 and 100\n"
            "- sentiment_data values should sum to ~100\n\n"
            f"Research:\n{research_text}"
        )

        response = client.chat(
            model="command-r-08-2024",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        content_text = response.message.content[0].text
        parsed = json.loads(content_text)

        state.region_data = parsed.get("region_data", [])[:5]
        state.age_data = parsed.get("age_data", [])
        state.sentiment_data = parsed.get("sentiment_data", {})

    except Exception as exc:
        state.error = f"Data extraction failed: {exc}"

    return state


def trend_synthesis_agent(state: MarketingState) -> MarketingState:
    """
    Synthesize cross-signal trend intelligence from existing research text and evidence.

    This node does NOT call external APIs. It only uses the already-populated
    research_results and evidence fields to infer structured trend_signals.
    """
    if state.error:
        return state

    import json

    try:
        if not state.research_results:
            state.trend_signals = []
            return state

        llm = ChatCohere(
            model="command-r-08-2024",
            cohere_api_key=state.cohere_api_key,
        )

        research_text = "\n\n".join(state.research_results)
        evidence_json = json.dumps(
            [e.model_dump() for e in state.evidence],
            ensure_ascii=False,
            indent=2,
        )

        prompt = (
            "You are a market intelligence analyst.\n\n"
            "Your task is to synthesize trend signals using THREE lenses:\n\n"
            "1. Reddit → Early emergence\n"
            "   - Look for repeated discussion topics\n"
            "   - Community curiosity, complaints, excitement\n"
            '   - Language like "anyone else noticing", "suddenly seeing"\n\n'
            "2. YouTube → Momentum\n"
            "   - Recent uploads\n"
            "   - Fast view growth\n"
            "   - Shorts or explainer formats appearing repeatedly\n\n"
            "3. News / Blogs → Narrative legitimacy\n"
            "   - Industry framing\n"
            "   - Economic, cultural, or regulatory context\n\n"
            "INPUT:\n"
            f"Marketing goal: {state.goal}\n\n"
            "Research text:\n"
            f"{research_text}\n\n"
            "EVIDENCE:\n"
            f"{evidence_json}\n\n"
            "OUTPUT RULES:\n"
            "- Produce 3–6 signals maximum\n"
            "- Each signal MUST be grounded in the research\n"
            "- Do NOT invent platforms or brands\n"
            "- Confidence must be between 0.0 and 1.0\n\n"
            "RETURN ONLY VALID JSON ARRAY:\n\n"
            "[\n"
            "  {\n"
            '    "source": "reddit | youtube | news",\n'
            '    "signal": "clear trend description",\n'
            '    "stage": "emerging | accelerating | established",\n'
            '    "confidence": 0.0,\n'
            '    "evidence": "short grounding explanation"\n'
            "  }\n"
            "]"
        )

        response = llm.invoke(prompt)
        raw_text = (
            response.content if isinstance(response.content, str) else str(response.content)
        )
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = "\n".join(cleaned.splitlines()[1:])

        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            state.trend_signals = parsed[:6]
        else:
            state.trend_signals = []

    except Exception:
        # Never raise; if synthesis fails, leave trend_signals empty.
        state.trend_signals = []

    return state


def visual_orchestrator(state: MarketingState) -> MarketingState:
    if state.error:
        return state

    import json

    try:
        client = cohere.ClientV2(api_key=state.cohere_api_key)

        prompt = (
            "Design 2–3 marketing charts.\n"
            "Each chart MUST have a stable id.\n\n"
            "Allowed ids:\n"
            "- age_distribution\n"
            "- regional_interest\n"
            "- sentiment_breakdown\n\n"
            "Return JSON:\n"
            "{\n"
            ' "charts": [\n'
            "   {\n"
            '     "id": "age_distribution",\n'
            '     "chart_type": "bar",\n'
            '     "title": "Interest by Age Group",\n'
            '     "labels": [],\n'
            '     "values": [],\n'
            '     "config": {"color_scheme": "blues", "orientation": "v"}\n'
            "   }\n"
            " ]\n"
            "}\n\n"
            f"Structured data:\n"
            f"Region: {state.region_data}\n"
            f"Age: {state.age_data}\n"
            f"Sentiment: {state.sentiment_data}"
        )

        response = client.chat(
            model="command-r-08-2024",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        content = response.message.content[0].text
        parsed = json.loads(content)
        state.visual_specs = parsed.get("charts", [])

    except Exception as exc:
        state.error = f"Visual orchestration failed: {exc}"

    return state


def analyst_agent(state: MarketingState) -> MarketingState:
    if state.error:
        return state

    if not state.visual_specs:
        state.error = "No visual specifications available to generate insights."
        return state

    import json

    try:
        llm = ChatCohere(
            model="command-r-08-2024",
            cohere_api_key=state.cohere_api_key,
        )

        goal_text = state.processed_goal or state.goal or ""
        charts_json = json.dumps(state.visual_specs, ensure_ascii=False, indent=2)
        evidence_json = json.dumps(
            [e.model_dump() for e in state.evidence],
            ensure_ascii=False,
            indent=2,
        )

        prompt = (
            "You are a senior marketing analyst.\n\n"
            f"GOAL:\n{goal_text}\n\n"
            "AVAILABLE CHARTS (index-based):\n"
            f"{charts_json}\n\n"
            "AVAILABLE EVIDENCE:\n"
            f"{evidence_json}\n\n"
            "RULES:\n"
            "- Generate at most ONE insight per chart (maximum 3 insights total).\n"
            "- Each insight MUST reference exactly one chart index via 'chart_ref'.\n"
            "- The 'source' field MUST reference one or more of the evidence items above.\n"
            "  Use their 'source' and/or part of their 'snippet' so the connection is clear.\n"
            "- No speculation, no recommendations, no advice. Only describe what the data shows.\n"
            "- Use precise, concrete language referencing real numbers or patterns from charts.\n\n"
            "OUTPUT JSON ONLY (no backticks, no prose):\n"
            "[\n"
            "  {\n"
            '    "text": "Clear, data-backed insight...",\n'
            '    "chart_ref": 0,\n'
            '    "source": "Tavily → Google Trends India (last 90 days)"\n'
            "  }\n"
            "]"
        )

        response = llm.invoke(prompt)
        raw_text = (
            response.content if isinstance(response.content, str) else str(response.content)
        )

        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = "\n".join(cleaned.splitlines()[1:])

        data = json.loads(cleaned)
        if not isinstance(data, list):
            raise ValueError("Expected a JSON array of insights.")

        insights: List[Insight] = []
        for item in data[:3]:
            item_json = json.dumps(item, ensure_ascii=False)
            insights.append(Insight.model_validate_json(item_json))

        state.insights = insights

    except Exception as exc:
        state.error = f"Analyst step failed: {exc}"

    return state


def scenario_simulator_agent(state: MarketingState) -> MarketingState:
    """Simulate relative outcomes for fixed alternate campaign scenarios."""
    if state.error:
        return state

    import json

    try:
        llm = ChatCohere(
            model="command-r-08-2024",
            cohere_api_key=state.cohere_api_key,
        )

        goal_text = state.processed_goal or state.goal or ""
        insights_json = json.dumps(
            [i.model_dump() for i in state.insights],
            ensure_ascii=False,
            indent=2,
        )
        age_json = json.dumps(state.age_data, ensure_ascii=False, indent=2)
        sentiment_json = json.dumps(state.sentiment_data, ensure_ascii=False, indent=2)
        competitors_json = json.dumps(
            state.competitor_signals,
            ensure_ascii=False,
            indent=2,
        )
        trend_signals_json = json.dumps(
            state.trend_signals,
            ensure_ascii=False,
            indent=2,
        )

        prompt = (
            "You are a marketing strategist simulating alternate campaign outcomes.\n\n"
            "Baseline campaign goal:\n"
            f"{goal_text}\n\n"
            "Audience age distribution:\n"
            f"{age_json}\n\n"
            "Audience sentiment:\n"
            f"{sentiment_json}\n\n"
            "Competitor activity:\n"
            f"{competitors_json}\n\n"
            "Cross-signal trend intelligence (Reddit / YouTube / News):\n"
            f"{trend_signals_json}\n\n"
            "Baseline insights (for context):\n"
            f"{insights_json}\n\n"
            "For EACH scenario below, estimate RELATIVE outcomes.\n"
            "Use trend_signals.stage and trend_signals.source as follows:\n"
            "- Emerging (often Reddit-heavy) → more exploratory positioning and testing.\n"
            "- Accelerating (often YouTube-heavy) → execution-ready formats and scaling.\n"
            "- Established (often News/Blogs-heavy) → authority and leadership framing.\n\n"
            "Estimate only RELATIVE outcomes (not numeric predictions):\n"
            "- Reach: Low | Medium | High\n"
            "- Sentiment: Negative | Neutral | Positive\n"
            "- Risk: Low | Medium | High\n\n"
            "Scenarios:\n"
            "1. Instagram – Gen Z\n"
            "2. LinkedIn – Millennials\n"
            "3. Delay launch by 2 weeks\n\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "scenarios": [\n'
            "    {\n"
            '      "scenario": "...",\n'
            '      "reach": "Low | Medium | High",\n'
            '      "sentiment": "Negative | Neutral | Positive",\n'
            '      "risk": "Low | Medium | High",\n'
            '      "reason": "Short explanation grounded in data"\n'
            "    }\n"
            "  ]\n"
            "}\n"
        )

        response = llm.invoke(prompt)
        raw_text = (
            response.content if isinstance(response.content, str) else str(response.content)
        )
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = "\n".join(cleaned.splitlines()[1:])

        parsed = json.loads(cleaned)
        scenarios = parsed.get("scenarios", [])
        state.scenario_results = scenarios if isinstance(scenarios, list) else []

    except Exception:
        # Fail gracefully: no scenarios rather than breaking the workflow.
        state.scenario_results = []

    return state


def strategy_mapper(state: MarketingState) -> MarketingState:
    """
    Map the current campaign and simulator context into platform-specific strategies.

    This node is intentionally local-only and does not call external APIs.
    It derives strategies for Reddit, YouTube, and News based on the synthesized
    cross-signal trends and existing simulator context.
    """
    if state.error:
        return state

    if not state.trend_signals:
        state.platform_strategies = []
        return state

    # Filter Reddit-only signals and pick the strongest.
    normalized: List[Dict[str, Any]] = []
    for ts in state.trend_signals:
        try:
            src = str(ts.get("source", "")).strip().lower()
            if src != "reddit":
                continue
            stage = str(ts.get("stage", "")).strip().lower()
            confidence = float(ts.get("confidence", 0.0))
            signal = str(ts.get("signal", "")).strip()
            evidence = str(ts.get("evidence", "")).strip()
            if not signal or not evidence or not (0.0 <= confidence <= 1.0):
                continue
            weight = 1.0
            if stage == "accelerating":
                weight = 1.5
            elif stage == "established":
                weight = 1.2
            normalized.append(
                {
                    "stage": stage,
                    "confidence": confidence,
                    "signal": signal,
                    "evidence": evidence,
                    "score": confidence * weight,
                }
            )
        except Exception:
            continue

    if not normalized:
        state.platform_strategies = []
        return state

    best = max(normalized, key=lambda s: s["score"])

    # Extract Reddit evidence from state.evidence (up to 5 threads).
    reddit_evs: List[Evidence] = []
    for ev in state.evidence:
        if "reddit" in (ev.source or "").lower():
            reddit_evs.append(ev)
    reddit_evs = reddit_evs[:5]

    reddit_threads: List[str] = []
    reddit_summaries: List[str] = []
    for ev in reddit_evs:
        if ev.source:
            reddit_threads.append(ev.source)
        if ev.snippet:
            reddit_summaries.append(ev.snippet[:200])

    strategies: List[Dict[str, Any]] = [
        {
            "platform": "Reddit",
            "why_fit": (
                f"Reddit discussions around \"{best['signal']}\" reveal what people "
                "are complaining about, excited by, and actively debating, making it "
                "a strong fit for early signal detection and exploratory testing."
            ),
            "content_pillar": (
                "Community threads, question-led posts, and feedback loops that "
                "lean into the specific pain points and curiosities found in the threads."
            ),
            "hashtags": [],
            "reddit_threads": reddit_threads,
            "reddit_summaries": reddit_summaries,
            "trend_reference": best["evidence"],
        }
    ]

    state.platform_strategies = strategies
    return state


# -----------------------------
# Campaign Strategy Agent (helper)
# -----------------------------


def generate_campaign_strategy(
    user_goal: str,
    social_media_trends: Any,
    reddit_discussions: Any,
    news_insights: Any,
    *,
    cohere_api_key: Optional[str] = None,
) -> str:
    """
    Campaign Strategy Agent.

    Generate a structured campaign plan using the existing Cohere configuration.
    This helper can be called from the main workflow after trend analysis.
    The response is formatted as markdown text with clearly labeled sections.
    """
    llm = ChatCohere(
        model="command-r-08-2024",
        cohere_api_key=cohere_api_key,
    )

    def _stringify(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        if isinstance(value, dict):
            return "\n".join(f"- {k}: {v}" for k, v in value.items())
        if isinstance(value, (list, tuple, set)):
            lines: List[str] = []
            for item in value:
                if isinstance(item, dict):
                    lines.append(
                        "- " + ", ".join(f"{k}={v}" for k, v in item.items())
                    )
                else:
                    lines.append(f"- {item}")
            return "\n".join(lines)
        return str(value)

    prompt = (
        "You are a senior marketing strategist.\n\n"
        "Use the information below to design a coherent, actionable campaign strategy.\n\n"
        f"USER GOAL:\n{user_goal or ''}\n\n"
        "SOCIAL MEDIA TREND INSIGHTS:\n"
        f"{_stringify(social_media_trends)}\n\n"
        "REDDIT DISCUSSION INSIGHTS:\n"
        f"{_stringify(reddit_discussions)}\n\n"
        "NEWS / BLOG INSIGHTS (if any):\n"
        f"{_stringify(news_insights)}\n\n"
        "RESPONSE REQUIREMENTS:\n"
        "- Be specific and grounded in the inputs above.\n"
        "- Do NOT hallucinate platforms, audiences, or budgets unrelated to the context.\n"
        "- Use clear, structured text that can be rendered directly in a Streamlit app.\n\n"
        "Return the strategy in the following sections, in this exact order:\n\n"
        "Campaign Theme\n"
        "- One or two sentences that describe the central campaign idea.\n\n"
        "Target Audience\n"
        "- Bullet points describing 2–4 plausible audience segments.\n\n"
        "Platform Selection\n"
        "- Bullet points listing which platforms to prioritize (e.g., Reddit, LinkedIn, Instagram, X, YouTube)\n"
        "  and why each is relevant to this goal.\n\n"
        "Content Ideas\n"
        "- 5–8 specific post ideas or content angles, each on its own bullet, referencing the trends and discussions where possible.\n\n"
        "Posting Schedule\n"
        "- A simple weekly schedule (e.g., days + times, frequency) tailored to the goal and platforms selected.\n\n"
        "Budget Suggestions\n"
        "- Suggest Low / Medium / High budget tiers.\n"
        "- For each tier, briefly describe what level of activity or reach it supports.\n\n"
        "Use concise markdown-style formatting with headings and bullet points.\n"
    )

    response = llm.invoke(prompt)
    text = response.content if isinstance(response.content, str) else str(
        response.content
    )
    return text.strip()


def creative_agent(state: MarketingState) -> MarketingState:
    if state.error:
        return state

    try:
        llm = ChatCohere(
            model="command-r-08-2024",
            cohere_api_key=state.cohere_api_key,
        )

        retriever = get_brand_voice_retriever(state.cohere_api_key)
        brand_docs = retriever.invoke("brand voice")
        brand_voice = "\n".join(d.page_content for d in brand_docs)

        insights_text = "\n".join(f"- {i.text}" for i in state.insights)

        prompt = (
            "Write a short social post.\n\n"
            f"Goal: {state.processed_goal}\n"
            f"Brand voice:\n{brand_voice}\n"
            f"Insights:\n{insights_text}\n"
        )

        response = llm.invoke(prompt)
        state.generated_post = response.content.strip()

    except Exception as exc:
        state.error = f"Creative step failed: {exc}"

    return state


# -----------------------------
# Graph builders
# -----------------------------

def build_marketing_graph() -> StateGraph:
    g = StateGraph(MarketingState)

    g.add_node("input_processor", input_processor)
    g.add_node("research_agent", research_agent)
    g.add_node("trend_synthesis_agent", trend_synthesis_agent)
    g.add_node("competitor_signal_agent", competitor_signal_agent)
    g.add_node("data_extractor", data_extractor)
    g.add_node("visual_orchestrator", visual_orchestrator)
    g.add_node("analyst_agent", analyst_agent)
    g.add_node("scenario_simulator_agent", scenario_simulator_agent)
    g.add_node("strategy_mapper", strategy_mapper)
    g.add_node("creative_agent", creative_agent)

    g.set_entry_point("input_processor")
    g.add_edge("input_processor", "research_agent")
    g.add_edge("research_agent", "trend_synthesis_agent")
    g.add_edge("trend_synthesis_agent", "competitor_signal_agent")
    g.add_edge("competitor_signal_agent", "data_extractor")
    g.add_edge("data_extractor", "visual_orchestrator")
    g.add_edge("visual_orchestrator", "analyst_agent")
    g.add_edge("analyst_agent", "scenario_simulator_agent")
    g.add_edge("scenario_simulator_agent", "strategy_mapper")
    g.add_edge("strategy_mapper", "creative_agent")
    g.add_edge("creative_agent", END)

    return g


def build_research_only_graph() -> StateGraph:
    g = StateGraph(MarketingState)

    g.add_node("input_processor", input_processor)
    g.add_node("research_agent", research_agent)
    g.add_node("trend_synthesis_agent", trend_synthesis_agent)
    g.add_node("competitor_signal_agent", competitor_signal_agent)
    g.add_node("data_extractor", data_extractor)
    g.add_node("visual_orchestrator", visual_orchestrator)
    g.add_node("analyst_agent", analyst_agent)
    g.add_node("scenario_simulator_agent", scenario_simulator_agent)
    g.add_node("strategy_mapper", strategy_mapper)

    g.set_entry_point("input_processor")
    g.add_edge("input_processor", "research_agent")
    g.add_edge("research_agent", "trend_synthesis_agent")
    g.add_edge("trend_synthesis_agent", "competitor_signal_agent")
    g.add_edge("competitor_signal_agent", "data_extractor")
    g.add_edge("data_extractor", "visual_orchestrator")
    g.add_edge("visual_orchestrator", "analyst_agent")
    g.add_edge("analyst_agent", "scenario_simulator_agent")
    g.add_edge("scenario_simulator_agent", "strategy_mapper")
    g.add_edge("strategy_mapper", END)

    return g


def build_creative_only_graph() -> StateGraph:
    g = StateGraph(MarketingState)
    g.add_node("creative_agent", creative_agent)
    g.set_entry_point("creative_agent")
    g.add_edge("creative_agent", END)
    return g


__all__ = [
    "MarketingState",
    "Insight",
    "build_marketing_graph",
    "build_research_only_graph",
    "build_creative_only_graph",
    "generate_campaign_strategy",
]