from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from graph import (
    MarketingState,
    build_creative_only_graph,
    build_research_only_graph,
    generate_campaign_strategy,
)


st.set_page_config(page_title="Social Marketing Assistant", layout="wide")


def _init_session_state() -> None:
    if "research_graph" not in st.session_state:
        st.session_state.research_graph = build_research_only_graph().compile()
    if "creative_graph" not in st.session_state:
        st.session_state.creative_graph = build_creative_only_graph().compile()
    if "state" not in st.session_state:
        st.session_state.state = MarketingState()


def _sidebar() -> None:
    with st.sidebar:
        st.title("Social Marketing Assistant")
        st.markdown("Configure your API keys.")

        cohere_key = st.text_input(
            "Cohere API Key",
            type="password",
            help="Used for Cohere chat and embeddings.",
        )
        tavily_key = st.text_input(
            "Tavily API Key",
            type="password",
            help="Used for trend research via Tavily.",
        )

        st.session_state.cohere_api_key = cohere_key or None
        st.session_state.tavily_api_key = tavily_key or None

        st.markdown("---")
        st.caption(
            "Tip: add a `brand_voice.txt` file next to `app.py` "
            "to customize your brand tone."
        )


def _run_research(graph: Any, goal: str) -> MarketingState:
    """Run research, extraction, analysis, and visual orchestration."""
    state = MarketingState(
        goal=goal,
        cohere_api_key=st.session_state.cohere_api_key,
        tavily_api_key=st.session_state.tavily_api_key,
    )

    with st.spinner("Researching trends and orchestrating visuals..."):
        with st.status("Running LangGraph workflow", expanded=True) as status:
            # Stream LangGraph execution so progress reflects node completion.
            # We rely on LangGraph to carry forward state; the last streamed
            # state is treated as the final state.
            final_raw: Any = state
            seen_nodes: set[str] = set()
            node_labels: Dict[str, str] = {
                "input_processor": "✓ Input processed.",
                "research_agent": "✓ Research collected.",
                "trend_synthesis_agent": "✓ Cross-signal trends synthesized.",
                "competitor_signal_agent": "✓ Competitor signals detected.",
                "data_extractor": "✓ Demographic and sentiment data extracted.",
                "visual_orchestrator": "✓ Charts and visuals designed.",
                "analyst_agent": "✓ Evidence-backed insights generated.",
                "scenario_simulator_agent": "✓ Campaign scenarios simulated.",
                "strategy_mapper": "✓ Platform strategies mapped.",
            }

            for event in graph.stream(state):
                # event is typically a dict mapping node_name -> state_update
                if isinstance(event, dict):
                    # UI feedback per completed node (once)
                    for node_name, node_state in event.items():
                        if node_name in node_labels and node_name not in seen_nodes:
                            st.write(node_labels[node_name])
                            seen_nodes.add(node_name)
                        # Track the most recent state-like object
                        final_raw = node_state
                else:
                    final_raw = event

            final_state: MarketingState = (
                MarketingState.model_validate(final_raw)
                if isinstance(final_raw, dict)
                else final_raw
            )

            if final_state.error:
                status.update(
                    label="Workflow finished with an error",
                    state="error",
                )
            else:
                status.update(
                    label="Research, analysis, and chart design complete",
                    state="complete",
                )

    return final_state


def _render_competitor_signals(competitor_signals: List[Dict[str, Any]]) -> None:
    if not competitor_signals:
        st.info("No competitor activity signals detected from the current research.")
        return

    for item in competitor_signals[:10]:
        competitor = str(item.get("competitor", "")).strip()
        signal = str(item.get("signal", "")).strip()
        confidence = item.get("confidence", None)
        reason = str(item.get("reason", "")).strip()

        sources = item.get("sources") or []
        if not isinstance(sources, list):
            sources = [str(sources)]

        cols = st.columns([2, 1, 1])
        with cols[0]:
            st.markdown(f"**{competitor or 'Unknown'}**")
        with cols[1]:
            st.markdown(f"**{signal or 'Unknown'}**")
        with cols[2]:
            try:
                pct = float(confidence) * 100.0
                st.markdown(f"**{pct:.0f}%**")
            except Exception:
                st.markdown("**—**")

        if reason:
            st.caption(reason)
        if sources:
            st.caption("Sources: " + ", ".join(str(s) for s in sources if str(s).strip()))


def _render_scenario_table(scenario_results: List[Dict[str, Any]]) -> None:
    if not scenario_results:
        st.info("Scenario simulator did not return results for this run.")
        return

    rows = []
    for s in scenario_results:
        rows.append(
            {
                "Scenario": s.get("scenario", ""),
                "Reach": s.get("reach", ""),
                "Sentiment": s.get("sentiment", ""),
                "Risk": s.get("risk", ""),
            }
        )

    df = pd.DataFrame(rows)
    st.caption("AI-simulated relative comparison — not a forecast")
    st.dataframe(df, use_container_width=True, hide_index=True)


def _run_creative(graph: Any, state: MarketingState) -> MarketingState:
    """Run just the creative step on an existing state."""
    state.cohere_api_key = st.session_state.cohere_api_key
    with st.spinner("Crafting a high-impact social post..."):
        with st.status("Running creative agent", expanded=True) as status:
            st.write("Generating post with brand voice RAG + insights.")

            raw_state = graph.invoke(state)
            final_state: MarketingState = (
                MarketingState.model_validate(raw_state)
                if isinstance(raw_state, dict)
                else raw_state
            )
            if final_state.error:
                status.update(
                    label="Creative step finished with an error",
                    state="error",
                )
            else:
                status.update(
                    label="Post generated successfully",
                    state="complete",
                )

    return final_state


def render_ai_charts(visual_specs: List[Dict[str, Any]], goal: str | None) -> None:
    """Render charts based on AI-generated visual specifications."""
    if not visual_specs:
        st.info("Run research to generate AI-selected charts.")
        return

    lower_goal = (goal or "").lower()
    num_charts = min(len(visual_specs), 3)
    cols = st.columns(num_charts)

    for idx, spec in enumerate(visual_specs[:num_charts]):
        chart_type = str(spec.get("chart_type", "bar")).lower()
        title = spec.get("title") or ""
        labels = spec.get("labels") or []
        values = spec.get("values") or []
        config = spec.get("config") or {}

        if not labels or not values or len(labels) != len(values):
            continue

        df = pd.DataFrame({"label": labels, "value": values})
        color_scheme = config.get("color_scheme", "blues")
        orientation = config.get("orientation", "v")

        with cols[idx]:
            fig = None

            if chart_type == "map":
                scope = "world"
                if "india" in lower_goal:
                    scope = "asia"
                fig = px.choropleth(
                    df,
                    locations="label",
                    locationmode="country names",
                    color="value",
                    color_continuous_scale=color_scheme.capitalize(),
                    scope=scope,
                )
            elif chart_type == "pie":
                fig = px.pie(
                    df,
                    names="label",
                    values="value",
                )
            elif chart_type == "line":
                fig = px.line(
                    df,
                    x="label",
                    y="value",
                )
            else:  # default to bar
                if orientation == "h":
                    fig = px.bar(
                        df,
                        x="value",
                        y="label",
                        orientation="h",
                    )
                else:
                    fig = px.bar(
                        df,
                        x="label",
                        y="value",
                        orientation="v",
                    )

            if fig is not None:
                fig.update_layout(
                    title=title,
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    _init_session_state()
    _sidebar()

    st.title("Social Marketing Assistant")
    st.markdown(
        "Automate **trend-aware** marketing posts using Cohere, Tavily, LangGraph, "
        "and AI-curated visualizations."
    )

    goal = st.text_input(
        "Marketing Goal",
        placeholder="e.g., Sell Samsung S26 to college students",
    )

    run_research = st.button(
        "Run Research & Design Charts",
        type="primary",
        disabled=not goal,
    )

    if run_research:
        if not st.session_state.cohere_api_key:
            st.error("Please enter your Cohere API key in the sidebar.")
        else:
            st.session_state.state = _run_research(
                st.session_state.research_graph,
                goal,
            )

    raw_state = st.session_state.state
    state: MarketingState = (
        MarketingState.model_validate(raw_state)
        if isinstance(raw_state, dict)
        else raw_state
    )

    if state.error:
        st.error(state.error)

    st.subheader("AI-Selected Visual Story")
    render_ai_charts(state.visual_specs, state.processed_goal)

    st.subheader("Key Insights")
    if state.insights:
        for idx, insight in enumerate(state.insights):
            st.markdown(f"- **Insight {idx + 1}**: {insight}")
    else:
        st.info("KPIs will appear here after research.")

    st.subheader("📱 Social Media Strategy")
    tabs = st.tabs(["Reddit"])

    strategies = state.platform_strategies or []
    if not strategies:
        for tab in tabs:
            with tab:
                st.info("Campaign strategies will appear here after research.")
    else:
        # Index strategies by platform for quick lookup
        by_platform: Dict[str, Dict[str, Any]] = {
            str(s.get("platform", "")).strip(): s for s in strategies
        }

        platform_order = ["Reddit"]
        for tab, platform in zip(tabs, platform_order):
            with tab:
                strat = by_platform.get(platform)
                if not strat:
                    st.info("Campaign strategies will appear here after research.")
                    continue

                st.markdown("🔥 What People Are Talking About")
                st.write(strat.get("why_fit", ""))

                st.markdown("🧵 Active Reddit Threads")
                threads = strat.get("reddit_threads") or []
                summaries = strat.get("reddit_summaries") or []
                for link, summary in zip(threads, summaries):
                    if link:
                        st.markdown(f"- [{link}]({link})")
                    if summary:
                        st.caption(summary)

                st.markdown("💡 How To Engage")
                st.write(strat.get("content_pillar", ""))

    st.subheader("Campaign Planning")
    # Automatically generate after research has produced insights
    can_generate_strategy = bool(state.insights) and not state.error
    if not can_generate_strategy:
        st.info("Campaign strategy will appear here after research.")
    elif not st.session_state.cohere_api_key:
        st.error("Please enter your Cohere API key in the sidebar.")
    else:
        user_goal = state.processed_goal or state.goal or ""
        social_media_trends = state.trend_signals
        reddit_discussions = [
            s for s in state.trend_signals if str(s.get("source", "")).lower() == "reddit"
        ]
        news_insights = [
            s for s in state.trend_signals if str(s.get("source", "")).lower() == "news"
        ]
        strategy_text = generate_campaign_strategy(
            user_goal=user_goal,
            social_media_trends=social_media_trends,
            reddit_discussions=reddit_discussions,
            news_insights=news_insights,
            cohere_api_key=st.session_state.cohere_api_key,
        )
        st.markdown(strategy_text)

    st.markdown("---")
    st.subheader("Content Preview")

    charts_ready = bool(state.visual_specs)
    can_create_post = charts_ready and bool(state.insights) and not state.error

    if charts_ready:
        create_post = st.button(
            "Create Post",
            disabled=not can_create_post,
        )
    else:
        create_post = False
        st.info("Run research first to generate charts, then you can create a post.")

    if create_post and can_create_post:
        if not st.session_state.cohere_api_key:
            st.error("Please enter your Cohere API key in the sidebar.")
        else:
            st.session_state.state = _run_creative(
                st.session_state.creative_graph,
                state,
            )
            raw_state = st.session_state.state
            state = (
                MarketingState.model_validate(raw_state)
                if isinstance(raw_state, dict)
                else raw_state
            )

    if state.generated_post:
        st.text_area(
            "Generated Post",
            value=state.generated_post,
            height=200,
        )
        st.button(
            "Copy to Clipboard (manual)",
            help="Select the text above and press Ctrl+C / Cmd+C.",
        )


if __name__ == "__main__":
    main()

