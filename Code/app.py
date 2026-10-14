
import hashlib
import re

import chromadb
import fitz
import streamlit as st
import os
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

# Load API credentials from the project-root .env file.
load_dotenv()

st.set_page_config(
    page_title="AutoDoc AI | Engineering Intelligence",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- DESIGN ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #F4F7FB;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #111C2E;
}

[data-testid="stSidebar"] * {
    color: #E2E8F0;
}

.brand {
    color: #FFFFFF;
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -0.8px;
}

.brand span {
    color: #5EEAD4;
}

.eyebrow {
    color: #0F766E;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
}

.hero-title {
    color: #172033 !important;
    font-size: 36px;
    font-weight: 700;
    line-height: 1.2;
    margin: 8px 0;
}

.hero-description {
    color: #526176 !important;
    font-size: 15px;
    line-height: 1.7;
    margin-bottom: 22px;
}

.section-heading {
    color: #172033 !important;
    font-size: 21px;
    font-weight: 700;
    margin-top: 14px;
}

.section-caption {
    color: #526176 !important;
    font-size: 13px;
}

.card {
    background: #FFFFFF;
    border: 1px solid #DFE7F0;
    border-radius: 14px;
    padding: 19px;
    margin-bottom: 14px;
}

.metric-label {
    color: #526176 !important;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.7px;
}

.metric-value {
    color: #172033 !important;
    font-size: 27px;
    font-weight: 700;
    margin-top: 7px;
}

.result-source {
    background: #E8F5F2;
    color: #115E59 !important;
    padding: 8px 11px;
    border-radius: 7px;
    font-size: 12px;
    font-weight: 600;
}


/* Consistent, readable button styling */
.stApp div.stButton > button,
.stApp div[data-testid="stDownloadButton"] > button {
    background-color: #0F766E !important;
    color: #FFFFFF !important;
    border: 1px solid #0F766E !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    min-height: 42px;
    opacity: 1 !important;
}

.stApp div.stButton > button p,
.stApp div[data-testid="stDownloadButton"] > button p {
    color: #FFFFFF !important;
}

.stApp div.stButton > button:hover,
.stApp div[data-testid="stDownloadButton"] > button:hover {
    background-color: #115E59 !important;
    color: #FFFFFF !important;
    border-color: #115E59 !important;
}

.stApp div.stButton > button:disabled,
.stApp div[data-testid="stDownloadButton"] > button:disabled {
    background-color: #E2E8F0 !important;
    color: #64748B !important;
    border-color: #CBD5E1 !important;
    opacity: 1 !important;
}


div[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1px dashed #91A4BA;
    border-radius: 12px;
    padding: 12px;
}

div[data-testid="stTextInput"] label,
div[data-testid="stTextArea"] label {
    color: #172033 !important;
}

div[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #DFE7F0;
    border-radius: 10px;
}

h1, h2, h3, h4, p, li {
    color: #172033;
}

#MainMenu, footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ---------- MODELS AND DATABASE ----------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_vector_database():
    return chromadb.PersistentClient(path="vector_store")


def make_chunks(pages, max_chars=850, overlap=120):
    chunks = []

    for page in pages:
        text = page["text"]
        if not text.strip():
            continue

        # Preserve paragraphs where possible.
        paragraphs = [
            p.strip() for p in re.split(r"\n\s*\n", text)
            if p.strip()
        ]

        current = ""

        for paragraph in paragraphs:
            if len(paragraph) > max_chars:
                if current:
                    chunks.append({
                        "page": page["page"],
                        "text": current.strip()
                    })
                    current = ""

                start = 0
                while start < len(paragraph):
                    part = paragraph[start:start + max_chars]
                    chunks.append({
                        "page": page["page"],
                        "text": part.strip()
                    })
                    start += max_chars - overlap
                continue

            if current and len(current) + len(paragraph) > max_chars:
                chunks.append({
                    "page": page["page"],
                    "text": current.strip()
                })
                current = current[-overlap:] + "\n" + paragraph
            else:
                current = (current + "\n" + paragraph).strip()

        if current.strip():
            chunks.append({
                "page": page["page"],
                "text": current.strip()
            })

    return chunks


def build_or_get_collection(pdf_bytes, pages):
    model = load_embedding_model()
    client = load_vector_database()

    file_hash = hashlib.sha256(pdf_bytes).hexdigest()[:16]
    collection_name = "hld_" + file_hash

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    if collection.count() == 0:
        chunks = make_chunks(pages)

        if chunks:
            texts = [c["text"] for c in chunks]
            embeddings = model.encode(
                texts,
                normalize_embeddings=True
            ).tolist()

            collection.add(
                ids=[f"chunk_{i}" for i in range(len(chunks))],
                documents=texts,
                embeddings=embeddings,
                metadatas=[
                    {"page": c["page"], "chunk": i}
                    for i, c in enumerate(chunks)
                ]
            )

    return collection, model


def generate_grounded_answer(question, passages):
    """Generate an answer using only retrieved PDF passages."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found. Check that your .env file is "
            "in the main AutoDoc-AI project folder and contains the key."
        )

    evidence = "\n\n".join(
        f"[Source {i}: PDF page {item['page']}]\n{item['text']}"
        for i, item in enumerate(passages, start=1)
    )
    prompt = f"""You are AutoDoc AI, an assistant for analysing automotive HLD documents.
Answer the user's question using ONLY the evidence passages below.
Rules:
- Do not use outside knowledge or invent specifications.
- If the evidence does not contain enough information, say: "The provided document does not contain enough evidence to answer this confidently."
- Cite supporting PDF pages inline, for example [Page 2]. Only cite page numbers shown in the evidence labels.
- Be concise and distinguish explicit document statements from any uncertainty.

USER QUESTION:
{question}

RETRIEVED EVIDENCE:
{evidence}
"""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
    )
    answer = (getattr(response, "text", None) or "").strip()
    if not answer:
        raise RuntimeError("Gemini returned an empty answer. Please try again.")
    return answer


# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown(
        '<div class="brand">AutoDoc <span>AI</span></div>'
        '<p style="color:#A7B5C9;font-size:10px;letter-spacing:1.5px">'
        'AUTOMOTIVE ENGINEERING</p>',
        unsafe_allow_html=True
    )

    st.divider()
    st.markdown("### Workspace")

    page = st.radio(
        "Navigation",
        ["Overview", "Document Search", "About Project"],
        label_visibility="collapsed"
    )

    st.divider()
    st.markdown("**PROJECT MODULES**")
    st.caption("01 · PDF extraction")
    st.caption("02 · Semantic retrieval")
    st.caption("03 · Grounded AI answers")
    st.caption("04 · Evaluation")

    st.divider()



# ---------- HEADER ----------
st.markdown(
    '<div class="eyebrow">ENGINEERING INTELLIGENCE / 01</div>'
    '<div class="hero-title">Architecture Document Analysis</div>'
    '<div class="hero-description">'
    'Explore automotive HLD documents and retrieve relevant evidence '
    'using natural-language search.'
    '</div>',
    unsafe_allow_html=True
)

# Session state stores extracted data for navigation between pages.
if "pages_data" not in st.session_state:
    st.session_state.pages_data = None
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None


# ---------- OVERVIEW ----------
if page == "Overview":
    st.markdown(
        '<div class="section-heading">Import your HLD document</div>'
        '<div class="section-caption">'
        'Upload a PDF to extract text and prepare it for semantic search.'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        "Choose an automotive HLD PDF",
        type=["pdf"],
        key="hld_upload"
    )

    if uploaded is not None:
        new_bytes = uploaded.getvalue()

        if new_bytes != st.session_state.pdf_bytes:
            try:
                with fitz.open(
                    stream=new_bytes,
                    filetype="pdf"
                ) as document:
                    pages_data = [
                        {
                            "page": number,
                            "text": p.get_text("text").strip()
                        }
                        for number, p in enumerate(document, start=1)
                    ]

                st.session_state.pdf_bytes = new_bytes
                st.session_state.pages_data = pages_data
                st.session_state.file_name = uploaded.name
                st.session_state.pop("collection", None)
                st.session_state.pop("indexed_hash", None)

            except Exception as exc:
                st.error(f"Unable to read PDF: {exc}")

    pages_data = st.session_state.pages_data

    if pages_data:
        total_pages = len(pages_data)
        total_chars = sum(len(p["text"]) for p in pages_data)
        pages_with_text = sum(bool(p["text"]) for p in pages_data)

        st.success(f"Processed: {st.session_state.file_name}")

        a, b, c = st.columns(3)

        for column, label, value in [
            (a, "TOTAL PAGES", total_pages),
            (b, "EXTRACTED CHARACTERS", f"{total_chars:,}"),
            (c, "PAGES WITH TEXT", f"{pages_with_text}/{total_pages}")
        ]:
            with column:
                st.markdown(
                    f'<div class="card">'
                    f'<div class="metric-label">{label}</div>'
                    f'<div class="metric-value">{value}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown(
            '<div class="section-heading">Extracted content</div>',
            unsafe_allow_html=True
        )

        filter_text = st.text_input(
            "Filter page text",
            placeholder="Try: Driver Assistance Controller"
        )

        for item in pages_data:
            if filter_text and filter_text.lower() not in item["text"].lower():
                continue

            with st.expander(f"Page {item['page']}"):
                if item["text"]:
                    st.text(item["text"])
                else:
                    st.warning(
                        "No selectable text found on this page; OCR may be needed."
                    )

        st.download_button(
            "Download extracted text",
            data="\n\n".join(
                f"PAGE {p['page']}\n{p['text']}" for p in pages_data
            ),
            file_name="extracted_hld_text.txt",
            mime="text/plain"
        )

        st.info(
            "Next: open Document Search in the sidebar to index the PDF "
            "and retrieve relevant passages."
        )

    else:
        st.markdown("""
        <div class="card">
            <h3>Start with your document</h3>
            <p style="color:#526176">
            Upload a permitted architecture PDF. AutoDoc AI will extract
            page-level text and prepare the document for semantic search.
            </p>
        </div>
        """, unsafe_allow_html=True)

        a, b, c = st.columns(3)
        for column, label, value in [
            (a, "DOCUMENT INGESTION", "Ready"),
            (b, "SEMANTIC SEARCH", "Next"),
            (c, "AI ANSWERS", "Planned")
        ]:
            with column:
                st.markdown(
                    f'<div class="card">'
                    f'<div class="metric-label">{label}</div>'
                    f'<div class="metric-value" style="font-size:21px">'
                    f'{value}</div></div>',
                    unsafe_allow_html=True
                )


# ---------- SEMANTIC SEARCH ----------
elif page == "Document Search":
    st.markdown(
        '<div class="section-heading">Search your document</div>'
        '<div class="section-caption">'
        'Find relevant passages by meaning, not just exact keywords.'
        '</div>',
        unsafe_allow_html=True
    )

    pages_data = st.session_state.pages_data
    pdf_bytes = st.session_state.pdf_bytes

    if not pages_data or not pdf_bytes:
        st.warning(
            "First upload an HLD PDF in the Overview section."
        )
    else:
        st.markdown(
            f'<div class="card"><b>Active document</b><br>'
            f'<span style="color:#526176">'
            f'{st.session_state.file_name}</span></div>',
            unsafe_allow_html=True
        )

        if st.button("Prepare document for semantic search"):
            with st.spinner(
                "Loading embedding model and indexing document. "
                "The first run may take a few minutes..."
            ):
                try:
                    collection, model = build_or_get_collection(
                        pdf_bytes, pages_data
                    )
                    st.session_state.collection = collection
                    st.session_state.indexed_hash = hashlib.sha256(
                        pdf_bytes
                    ).hexdigest()[:16]
                    st.success(
                        f"Index ready: {collection.count()} text chunks."
                    )
                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")

        collection = st.session_state.get("collection")

        if collection is not None:
            st.success(
                f"Search index ready with {collection.count()} chunks."
            )

            question = st.text_input(
                "What do you want to find?",
                placeholder=(
                    "Example: Which component evaluates obstacle detection?"
                )
            )

            top_k = st.slider(
                "Number of passages to retrieve",
                min_value=1,
                max_value=5,
                value=3
            )

            if st.button("Search document"):
                if not question.strip():
                    st.warning("Enter a question first.")
                else:
                    with st.spinner("Searching document passages..."):
                        try:
                            model = load_embedding_model()
                            query_vector = model.encode(
                                [question],
                                normalize_embeddings=True
                            ).tolist()

                            results = collection.query(
                                query_embeddings=query_vector,
                                n_results=min(top_k, collection.count()),
                                include=[
                                    "documents",
                                    "metadatas",
                                    "distances"
                                ]
                            )

                            # Build the exact evidence bundle used by Gemini.
                            retrieved_passages = []
                            for i, text in enumerate(results["documents"][0]):
                                metadata = results["metadatas"][0][i]
                                distance = results["distances"][0][i]
                                retrieved_passages.append({
                                    "page": int(metadata["page"]),
                                    "text": text,
                                    "distance": float(distance),
                                })

                            st.markdown(
                                '<div class="section-heading">'
                                'AI-generated answer</div>',
                                unsafe_allow_html=True
                            )
                            with st.container(border=True):
                                with st.spinner("Generating an evidence-grounded answer..."):
                                    try:
                                        answer = generate_grounded_answer(
                                            question, retrieved_passages
                                        )
                                        st.markdown(answer)
                                    except Exception as exc:
                                        st.error(
                                            "Could not generate an answer. "
                                            "Check your API key, internet connection, "
                                            "and Gemini API quota, then try again."
                                        )
                                        st.caption(f"Technical detail: {exc}")

                            st.markdown(
                                '<div class="section-heading">'
                                'Retrieved evidence</div>',
                                unsafe_allow_html=True
                            )
                            st.caption(
                                "Review these source passages to verify the answer. "
                                "Page citations refer to the uploaded PDF."
                            )

                            for i, item in enumerate(retrieved_passages, start=1):
                                with st.container(border=True):
                                    st.markdown(f"**Source passage {i}**")
                                    st.markdown(
                                        f'<div class="result-source">'
                                        f'Source: Page {item["page"]}</div>',
                                        unsafe_allow_html=True
                                    )
                                    st.write(item["text"])
                                    st.caption(
                                        f"Cosine distance: {item['distance']:.4f} "
                                        "(lower is closer in embedding space)"
                                    )

                        except Exception as exc:
                            st.error(f"Search failed: {exc}")


# ---------- ABOUT ----------
elif page == "About Project":
    st.markdown("### About AutoDoc AI")

    st.write(
        "AutoDoc AI is a student prototype for analysing automotive "
        "High-Level Design documents."
    )

    st.markdown("#### Implemented modules")
    st.write(
        "1. PDF upload and page-aware text extraction.\n"
        "2. Semantic chunking and embedding generation.\n"
        "3. ChromaDB storage and semantic retrieval."
    )

    st.markdown("#### AI answer generation")
    st.write(
        "Gemini generates answers from retrieved PDF passages only. "
        "The interface displays source passages and asks the model to cite "
        "the PDF page numbers supplied with that evidence."
    )

    st.warning(
        "This educational prototype does not approve architecture "
        "decisions or replace qualified engineering review."
    )
