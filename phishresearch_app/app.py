
import os
import time

import pandas as pd
import streamlit as st

import retrieval_engine
import llm_engine


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="PhishResearch AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)

METRICS_PATH = os.path.join(
    BASE_DIR, "retrieval_metrics_summary.csv"
)

CHUNKS_PATH = os.path.join(
    BASE_DIR,
    "saved_retrieval_artifacts",
    "retrieval_chunks.csv"
)


# =====================================================
# 2. CUSTOM UI DESIGN
# =====================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background: #f4f7fb;
    color: #172b4d;
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101d35 0%, #172d50 100%);
}

section[data-testid="stSidebar"] * {
    color: #ffffff;
}

/* Sidebar radio buttons */
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: rgba(255,255,255,0.04);
    padding: 10px 12px;
    border-radius: 8px;
    margin-bottom: 5px;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.12);
}

/* Titles */
.main-heading {
    font-size: 34px;
    font-weight: 800;
    color: #142747;
    letter-spacing: -0.8px;
}

.page-subtitle {
    color: #66758c;
    font-size: 15px;
    margin-top: -8px;
    margin-bottom: 25px;
}

/* Section headings */
.section-heading {
    font-size: 21px;
    font-weight: 750;
    color: #172b4d;
    margin-top: 12px;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e3eaf3;
    border-radius: 13px;
    padding: 18px;
    box-shadow: 0 3px 12px rgba(20, 39, 71, 0.035);
}

div[data-testid="stMetricLabel"] {
    color: #66758c;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #173b70;
    font-weight: 800;
}

/* Buttons */
div.stButton > button,
div.stDownloadButton > button {
    background: #1d4ed8;
    color: #ffffff;
    border: none;
    border-radius: 9px;
    padding: 10px 18px;
    font-weight: 650;
    transition: 0.2s ease;
}

div.stButton > button:hover,
div.stDownloadButton > button:hover {
    background: #173ea8;
    color: #ffffff;
    border: none;
}

/* Input fields */
.stTextArea textarea,
.stTextInput input,
.stSelectbox div[data-baseweb="select"] {
    border-radius: 9px;
}

/* Cards */
.info-card {
    background: #ffffff;
    border: 1px solid #e3eaf3;
    border-radius: 13px;
    padding: 19px;
    margin-bottom: 12px;
}

/* Answer panel */
.answer-panel {
    background: #ffffff;
    border: 1px solid #dce6f3;
    border-left: 5px solid #2563eb;
    border-radius: 12px;
    padding: 22px;
    margin-top: 12px;
    margin-bottom: 18px;
}

/* Reference panel */
.reference-panel {
    background: #ffffff;
    border: 1px solid #dce6f3;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 10px;
}

/* Small labels */
.small-muted {
    color: #718096;
    font-size: 13px;
}

/* Divider */
hr {
    border-color: #e2e8f0;
}

/* Expanders */
div[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #e3eaf3;
    border-radius: 10px;
}

/* Tables */
div[data-testid="stDataFrame"] {
    border: 1px solid #e3eaf3;
    border-radius: 10px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. SESSION STATE
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_results" not in st.session_state:
    st.session_state.last_results = None

if "last_question" not in st.session_state:
    st.session_state.last_question = ""

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = {}


# =====================================================
# 4. HELPER FUNCTIONS
# =====================================================

@st.cache_data
def load_corpus_metadata():

    if not os.path.exists(CHUNKS_PATH):
        return None

    df = pd.read_csv(CHUNKS_PATH)

    columns = [
        "Article_ID",
        "Title",
        "Year"
    ]

    available = [
        c for c in columns
        if c in df.columns
    ]

    return df[available].drop_duplicates(
        subset=["Article_ID"]
    )


def get_engine():
    return retrieval_engine.load_engine()


def run_retrieval(query, method, top_k):

    (
        chunks,
        embeddings,
        bm25,
        embedding_model,
        cross_encoder
    ) = get_engine()

    return retrieval_engine.retrieve(
        query=query,
        method=method,
        top_k=top_k,
        chunks=chunks,
        bm25=bm25,
        embeddings=embeddings,
        embedding_model=embedding_model,
        cross_encoder=cross_encoder
    )


def safe_value(item, column, default="Not available"):

    value = item.get(column, default)

    if pd.isna(value):
        return default

    return str(value)


def create_reference_list(results):

    """
    Create a reference list directly from retrieved evidence.
    These references are not generated by the language model.
    """

    if results is None or results.empty:
        return pd.DataFrame()

    reference_columns = [
        "Article_ID",
        "Title",
        "Year",
        "Pages",
        "Chunk_ID"
    ]

    available_columns = [
        column for column in reference_columns
        if column in results.columns
    ]

    references = results[available_columns].copy()

    # Remove duplicate passages with the same chunk ID
    if "Chunk_ID" in references.columns:
        references = references.drop_duplicates(
            subset=["Chunk_ID"]
        )

    references = references.reset_index(drop=True)

    references.insert(
        0,
        "Reference",
        [f"[{i + 1}]" for i in range(len(references))]
    )

    return references


def show_references(results):

    """
    Display accessible references and supporting passages.
    Each reference is linked to the retrieved evidence in the UI.
    """

    if results is None or results.empty:
        st.info("No references are available.")
        return

    st.markdown("## 📑 Sources and References")

    st.caption(
        "The following references are taken directly from the "
        "retrieved research passages. Expand a reference to inspect "
        "its supporting text."
    )

    references = create_reference_list(results)

    if references.empty:
        st.warning(
            "Reference metadata could not be generated "
            "from the retrieved results."
        )
        return

    for i, (_, item) in enumerate(results.iterrows()):

        reference_number = i + 1

        article_id = safe_value(item, "Article_ID")
        title = safe_value(item, "Title")
        year = safe_value(item, "Year")
        pages = safe_value(item, "Pages")
        chunk_id = safe_value(item, "Chunk_ID")
        section = safe_value(item, "Section")
        passage = safe_value(item, "Text", "No passage text available.")

        with st.expander(
            f"[{reference_number}] {title} — {year}"
        ):

            st.markdown(
                f"**Article ID:** {article_id}"
            )

            st.markdown(
                f"**Title:** {title}"
            )

            st.markdown(
                f"**Publication year:** {year}"
            )

            st.markdown(
                f"**Page(s):** {pages}"
            )

            st.markdown(
                f"**Section:** {section}"
            )

            st.markdown(
                f"**Chunk ID:** `{chunk_id}`"
            )

            st.markdown("**Supporting research passage:**")

            st.write(passage)

    st.download_button(
        label="⬇️ Download reference list (CSV)",
        data=references.to_csv(index=False).encode("utf-8"),
        file_name="phishresearch_references.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"download_references_{id(results)}"
    )


def show_evidence(results):

    if results is None or results.empty:
        st.info("No evidence passages are available.")
        return

    st.markdown("### 🔍 Retrieved Research Evidence")

    st.write(
        f"Retrieved **{len(results)} research passages**."
    )

    display_columns = [
        "Article_ID",
        "Title",
        "Year",
        "Pages",
        "Section"
    ]

    display_columns = [
        c for c in display_columns
        if c in results.columns
    ]

    st.dataframe(
        results[display_columns],
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Inspect Retrieved Passages")

    for i, (_, item) in enumerate(results.iterrows()):

        article_id = safe_value(item, "Article_ID")
        title = safe_value(item, "Title")
        pages = safe_value(item, "Pages")
        year = safe_value(item, "Year")
        chunk_id = safe_value(item, "Chunk_ID")

        with st.expander(
            f"Passage {i + 1}: {article_id} — {title}"
        ):

            st.caption(
                f"Year: {year} | "
                f"Pages: {pages} | "
                f"Chunk ID: {chunk_id}"
            )

            st.write(
                safe_value(item, "Text", "No passage text available.")
            )

    st.download_button(
        label="⬇️ Download retrieved evidence (CSV)",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="retrieved_research_evidence.csv",
        mime="text/csv",
        key=f"download_evidence_{id(results)}"
    )


def show_answer_and_references():

    if st.session_state.last_answer:

        st.markdown(
            '<div class="section-heading">🤖 AI Research Answer</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="answer-panel">',
            unsafe_allow_html=True
        )

        st.markdown(
            st.session_state.last_answer
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        st.info(
            "Check the source references below against the supporting "
            "passages. The reference list is generated from retrieved "
            "corpus records; it does not independently verify every "
            "claim made by Qwen."
        )

        show_references(
            st.session_state.last_results
        )


# =====================================================
# 5. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.markdown(
    """
    <div style="padding: 12px 0 20px 0;">
        <div style="font-size: 29px; font-weight: 850;">
            🛡️ PhishResearch AI
        </div>
        <div style="font-size: 13px; opacity: 0.75;">
            Evidence-Grounded Research Assistant
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Research Assistant",
        "Evidence Explorer",
        "Retrieval Comparison",
        "Evaluation Dashboard",
        "Research Corpus",
        "System Information"
    ]
)

st.sidebar.divider()

st.sidebar.markdown(
    "**Research domain**"
)

st.sidebar.caption(
    "Phishing and malicious threat detection"
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Local language model: Qwen2.5-1.5B-Instruct"
)


# =====================================================
# 6. DASHBOARD
# =====================================================

if page == "Dashboard":

    st.markdown(
        '<div class="main-heading">Research Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Explore phishing research through intelligent retrieval '
        'and evidence-grounded question answering.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Research Papers", "184")
    col2.metric("Indexed Passages", "5,213")
    col3.metric("Embedding Dimensions", "384")
    col4.metric("Evaluation Questions", "80")

    st.divider()

    st.markdown(
        '<div class="section-heading">System Capabilities</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:

        st.markdown("""
        <div class="info-card">
            <h3>🔎 Information Retrieval</h3>
            <p>Retrieve relevant research passages using:</p>
            <ul>
                <li>BM25 keyword retrieval</li>
                <li>SBERT semantic retrieval</li>
                <li>Hybrid Reciprocal Rank Fusion</li>
                <li>Cross-Encoder reranking</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with right:

        st.markdown("""
        <div class="info-card">
            <h3>🤖 Research Assistance</h3>
            <p>Explore the research corpus through:</p>
            <ul>
                <li>Local Qwen answer generation</li>
                <li>Article and page-level references</li>
                <li>Passage-level evidence inspection</li>
                <li>Downloadable evidence and references</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.subheader("Research Workflow")

    workflow_cols = st.columns(5)

    workflow_steps = [
        ("01", "Ask", "Enter a research question"),
        ("02", "Retrieve", "Find relevant passages"),
        ("03", "Rerank", "Rank evidence by relevance"),
        ("04", "Generate", "Generate a Qwen answer"),
        ("05", "Verify", "Inspect sources and pages")
    ]

    for col, (number, heading, description) in zip(
        workflow_cols, workflow_steps
    ):

        with col:

            st.markdown(
                f"""
                <div class="info-card">
                    <div style="color:#2563eb;font-weight:800;">
                        {number}
                    </div>
                    <h4>{heading}</h4>
                    <div class="small-muted">
                        {description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.info(
        "Start from Research Assistant to retrieve research evidence "
        "and generate an answer using the local Qwen model."
    )


# =====================================================
# 7. RESEARCH ASSISTANT
# =====================================================

elif page == "Research Assistant":

    st.markdown(
        '<div class="main-heading">🔎 Research Assistant</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Ask questions about phishing detection and explore '
        'supporting evidence from the research corpus.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("### What would you like to research?")

    st.caption(
        "Example: How effective are machine learning models in "
        "detecting phishing websites?"
    )

    question = st.chat_input(
        "Ask PhishResearch AI anything about phishing detection..."
    )

    col1, col2 = st.columns(2)

    with col1:

        method = st.selectbox(
            "Retrieval method",
            [
                "BM25",
                "SBERT",
                "Hybrid RRF",
                "Hybrid RRF + Cross-Encoder"
            ],
            index=3
        )

    with col2:

        top_k = st.selectbox(
            "Number of evidence passages",
            [3, 5, 10, 20],
            index=1
        )

    # -------------------------------------------------
    # RETRIEVE RESEARCH EVIDENCE
    # -------------------------------------------------

    if question:

        try:

            with st.spinner(
                "Searching the research corpus..."
            ):

                start = time.time()

                results = run_retrieval(
                    question.strip(),
                    method,
                    top_k
                )

                elapsed = time.time() - start

            st.session_state.last_results = results

            st.session_state.last_question = question.strip()

            st.session_state.last_answer = ""

            st.session_state.history.append({
                "Question": question.strip(),
                "Method": method,
                "Passages": len(results),
                "Time (seconds)": round(elapsed, 2)
            })

            st.success(
                f"Successfully retrieved {len(results)} passages "
                f"in {elapsed:.2f} seconds."
            )

        except Exception as e:

            st.error("Research retrieval failed.")

            st.exception(e)

    # -------------------------------------------------
    # DISPLAY RETRIEVED EVIDENCE
    # -------------------------------------------------

    if st.session_state.last_results is not None:

        st.divider()

        with st.chat_message("user"):
            st.markdown(st.session_state.last_question)

        show_evidence(
            st.session_state.last_results
        )

        st.divider()

        st.markdown(
            '<div class="section-heading">🤖 Generate AI Research Answer</div>',
            unsafe_allow_html=True
        )

        st.write(
            "Generate an answer using the retrieved research passages "
            "and the local Qwen2.5-1.5B-Instruct model."
        )

        if st.button(
            "✨ Generate AI Answer with Qwen",
            use_container_width=True
        ):

            if st.session_state.last_results.empty:

                st.warning(
                    "No evidence is available for answer generation."
                )

            else:

                try:

                    st.session_state.last_answer = ""

                    with st.spinner(
                        "Qwen is analysing the retrieved research evidence..."
                    ):

                        answer = llm_engine.generate_answer(
                            question=st.session_state.last_question,
                            evidence_df=st.session_state.last_results,
                            use_retrieval=True,
                            max_new_tokens=400
                        )

                    st.session_state.last_answer = answer

                    st.success(
                        "Qwen answer generation completed."
                    )

                except Exception as e:

                    st.error(
                        "Qwen answer generation failed."
                    )

                    st.exception(e)

        if st.session_state.last_answer:

            st.divider()

            with st.chat_message("assistant"):
                show_answer_and_references()


# =====================================================
# 8. EVIDENCE EXPLORER
# =====================================================

elif page == "Evidence Explorer":

    st.markdown(
        '<div class="main-heading">📚 Evidence Explorer</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Inspect retrieved passages, source information, '
        'and supporting research text.'
        '</div>',
        unsafe_allow_html=True
    )

    if st.session_state.last_results is not None:

        show_evidence(
            st.session_state.last_results
        )

        st.divider()

        show_references(
            st.session_state.last_results
        )

    else:

        st.info(
            "No retrieved evidence yet. "
            "Go to Research Assistant and submit a question first."
        )


# =====================================================
# 9. RETRIEVAL COMPARISON
# =====================================================

elif page == "Retrieval Comparison":

    st.markdown(
        '<div class="main-heading">⚖️ Retrieval Comparison</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Compare the evidence returned by the four retrieval methods.'
        '</div>',
        unsafe_allow_html=True
    )

    comparison_question = st.text_area(
        "Comparison question",
        value=(
            "What machine learning techniques are used "
            "for phishing detection?"
        ),
        height=100
    )

    comparison_k = st.selectbox(
        "Passages per method",
        [3, 5, 10],
        index=1
    )

    if st.button(
        "▶ Run Retrieval Comparison",
        use_container_width=True
    ):

        methods = [
            "BM25",
            "SBERT",
            "Hybrid RRF",
            "Hybrid RRF + Cross-Encoder"
        ]

        if not comparison_question.strip():

            st.warning("Enter a comparison question.")

        else:

            comparison_data = {}

            try:

                for current_method in methods:

                    with st.spinner(
                        f"Running {current_method}..."
                    ):

                        start = time.time()

                        current_results = run_retrieval(
                            comparison_question.strip(),
                            current_method,
                            comparison_k
                        )

                        elapsed = time.time() - start

                    comparison_data[current_method] = {
                        "results": current_results,
                        "time": round(elapsed, 2)
                    }

                st.session_state.comparison_results = comparison_data

                st.success(
                    "All four retrieval methods completed."
                )

            except Exception as e:

                st.error("Retrieval comparison failed.")

                st.exception(e)

    if st.session_state.comparison_results:

        st.divider()

        st.subheader("Comparison Summary")

        summary = []

        for method_name, item in (
            st.session_state.comparison_results.items()
        ):

            summary.append({
                "Method": method_name,
                "Retrieved passages": len(item["results"]),
                "Execution time (seconds)": item["time"]
            })

        summary_df = pd.DataFrame(summary)

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            summary_df.set_index("Method")[
                "Execution time (seconds)"
            ]
        )

        st.subheader("Retrieved Evidence by Method")

        for method_name, item in (
            st.session_state.comparison_results.items()
        ):

            with st.expander(
                f"📄 {method_name}"
            ):

                show_evidence(
                    item["results"]
                )

                show_references(
                    item["results"]
                )


# =====================================================
# 10. EVALUATION DASHBOARD
# =====================================================

elif page == "Evaluation Dashboard":

    st.markdown(
        '<div class="main-heading">📊 Evaluation Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Review retrieval performance using the saved evaluation metrics.'
        '</div>',
        unsafe_allow_html=True
    )

    if os.path.exists(METRICS_PATH):

        metrics = pd.read_csv(METRICS_PATH)

        st.success(
            "Saved retrieval evaluation metrics loaded."
        )

        st.dataframe(
            metrics,
            use_container_width=True,
            hide_index=True
        )

        if "Method" in metrics.columns:

            numeric_columns = [
                c for c in [
                    "Mean Recall@5",
                    "Mean MRR@10",
                    "Mean nDCG@10"
                ]
                if c in metrics.columns
            ]

            if numeric_columns:

                chart_data = metrics.set_index("Method")[
                    numeric_columns
                ]

                st.subheader(
                    "Retrieval Performance Comparison"
                )

                st.bar_chart(chart_data)

        st.caption(
            "Interpret these results using the evaluation coverage "
            "and manual relevance-judgment limitations documented "
            "in the technical report."
        )

    else:

        st.warning(
            "The retrieval metrics summary CSV was not found "
            "at the expected project location."
        )


# =====================================================
# 11. RESEARCH CORPUS
# =====================================================

elif page == "Research Corpus":

    st.markdown(
        '<div class="main-heading">📖 Research Corpus</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Explore the article metadata available in the indexed corpus.'
        '</div>',
        unsafe_allow_html=True
    )

    corpus = load_corpus_metadata()

    if corpus is not None:

        col1, col2 = st.columns(2)

        col1.metric(
            "Unique Research Papers",
            corpus["Article_ID"].nunique()
        )

        if "Year" in corpus.columns:

            col2.metric(
                "Publication Years",
                corpus["Year"].nunique()
            )

        search_text = st.text_input(
            "Search by article ID or title",
            placeholder="Enter an article ID or title..."
        )

        filtered = corpus.copy()

        if search_text.strip():

            mask = (
                filtered["Article_ID"].astype(str).str.contains(
                    search_text,
                    case=False,
                    na=False
                )
                |
                filtered["Title"].astype(str).str.contains(
                    search_text,
                    case=False,
                    na=False
                )
            )

            filtered = filtered[mask]

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            label="⬇️ Download corpus metadata",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="research_corpus_metadata.csv",
            mime="text/csv"
        )

    else:

        st.warning(
            "Could not find the retrieval chunks CSV. "
            "Check the saved_retrieval_artifacts folder."
        )


# =====================================================
# 12. SYSTEM INFORMATION
# =====================================================

elif page == "System Information":

    st.markdown(
        '<div class="main-heading">⚙️ System Information</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-subtitle">'
        'Technical components and workflow of PhishResearch AI.'
        '</div>',
        unsafe_allow_html=True
    )

    system_info = pd.DataFrame([
        ["Dense embedding model", "all-MiniLM-L6-v2"],
        ["Embedding dimensions", "384"],
        ["Keyword retrieval", "BM25Okapi"],
        [
            "Cross-Encoder",
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ],
        ["Hybrid fusion", "Reciprocal Rank Fusion"],
        ["Language model", "Qwen2.5-1.5B-Instruct"],
        ["Indexed passages", "5,213"],
        ["Research papers", "184"],
        ["Application framework", "Streamlit"]
    ], columns=["Component", "Configuration"])

    st.dataframe(
        system_info,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Research Assistant Workflow")

    st.code(
        "Research question\n"
        "       ↓\n"
        "BM25 + SBERT retrieval\n"
        "       ↓\n"
        "Reciprocal Rank Fusion\n"
        "       ↓\n"
        "Cross-Encoder reranking\n"
        "       ↓\n"
        "Ranked research evidence\n"
        "       ↓\n"
        "Qwen2.5-1.5B-Instruct\n"
        "       ↓\n"
        "Generated research answer\n"
        "       ↓\n"
        "Accessible source references\n"
        "       ↓\n"
        "Evidence verification"
    )

    st.info(
        "References displayed in the application are constructed "
        "from the retrieved corpus records. Users should verify that "
        "each generated claim is supported by its cited passage."
    )


# =====================================================
# 13. FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#718096;
        font-size:13px;
        padding:12px;
    ">
        <b>PhishResearch AI</b><br>
        Evidence-Grounded Research Assistant<br>
        Academic Research Project
    </div>
    """,
    unsafe_allow_html=True
)
