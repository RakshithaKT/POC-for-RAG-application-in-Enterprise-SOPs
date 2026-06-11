import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from groq import Groq
import base64
import os
import json
import random
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Multimodal RAG for Enterprise SOPs",
    page_icon="🔬",
    layout="wide"
)


CHROMA_PATH      = "./chroma_db"
COLLECTION_NAME  = "research_papers_multimodal"
IMAGES_FOLDER    = os.path.join(".", "paper_images")
VISION_MODEL     = "meta-llama/llama-4-scout-17b-16e-instruct"
JUDGE_MODEL      = "llama-3.3-70b-versatile"   

def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def resolve_image_path(uri: str) -> str:
    """
    The URI stored in ChromaDB may be an absolute path (written by ingest.py)
    or a relative path.  Fall back to IMAGES_FOLDER + basename if needed.
    """
    if os.path.exists(uri):
        return uri
    return os.path.join(IMAGES_FOLDER, os.path.basename(uri))


def parse_json_response(raw: str) -> dict:
    """Strip markdown code-fences if the model wraps its JSON in them."""
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


@st.cache_resource
def init_chroma():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_function = OpenCLIPEmbeddingFunction()
    image_loader = ImageLoader()
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        data_loader=image_loader,
    )

st.sidebar.header("Configuration")

groq_api_key = os.getenv("GROQ_API_KEY")
groq_client  = Groq() if groq_api_key else None

if not groq_api_key:
    st.sidebar.warning("GROQ_API_KEY not found in .env — LLM features disabled.")

try:
    collection = init_chroma()
    st.sidebar.success(f"Connected to ChromaDB!  Pages indexed: {collection.count()}")
except Exception as e:
    st.sidebar.error(
        f"Failed to load ChromaDB. "
        f"Ensure the 'chroma_db' folder is in your project root. Error: {e}"
    )
    st.stop()   


st.title("🔬 Multimodal Research Assistant")
st.write("Query your research papers visually using OpenCLIP embeddings and Llama 4 Scout via Groq.")


tab_query, tab_eval = st.tabs(["🔍 Query", "📊 Evaluate"])


with tab_query:
    user_query = st.text_input(
        "Ask a question about your research papers "
        "(e.g., 'Show me the graph on revenue trends'):"
    )

    if user_query:
        if not groq_api_key:
            st.error("Please set GROQ_API_KEY in your .env file before running a query.")
        else:
            with st.spinner("Searching the vector database..."):
                results = collection.query(
                    query_texts=[user_query],
                    n_results=5,
                    include=["uris", "metadatas"],
                )

            if results and results["ids"][0]:
                ids       = results["ids"][0]
                uris      = results["uris"][0]
                metadatas = results["metadatas"][0]

                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader(" Retrieved Context Pages")
                    for i in range(len(ids)):
                        st.caption(
                            f"**Match {i+1}:** "
                            f"{metadatas[i].get('source_paper')} "
                            f"(Page {metadatas[i].get('page_number')})"
                        )
                        local_uri = resolve_image_path(uris[i])
                        try:
                            st.image(Image.open(local_uri), use_container_width=True)
                        except Exception as img_err:
                            st.error(f"Could not render image: {img_err}")

                with col2:
                    st.subheader("🤖 LLM Analysis")
                    with st.spinner("Llama 4 Scout is analyzing all retrieved pages..."):
                        try:
                            prompt_content = [{
                                "type": "text",
                                "text": (
                                    f"Analyze these document images and answer the following question: "
                                    f"{user_query}. "
                                    f"Be concise and factually stick ONLY to what is visible in these pages."
                                ),
                            }]
                            for uri in uris:
                                local_uri = resolve_image_path(uri)
                                prompt_content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,"
                                               f"{encode_image_base64(local_uri)}"
                                    },
                                })

                            response = groq_client.chat.completions.create(
                                model=VISION_MODEL,
                                messages=[{"role": "user", "content": prompt_content}],
                                temperature=0.1,
                            )
                            st.markdown(response.choices[0].message.content)

                        except Exception as api_err:
                            st.error(f"Groq API Error: {api_err}")
            else:
                st.warning("No relevant matching pages found in the database.")



with tab_eval:
    st.subheader("📊 LLM-as-Judge Evaluation")
#     st.markdown("""
# Automatically evaluates the RAG pipeline — no manual labelling needed.


# 1. Samples random pages from the database
# 2. Uses the vision LLM to generate a Q&A pair per page — this becomes the ground truth
# 3. Runs each question through the RAG pipeline exactly as the Query tab would
# 4. A separate judge LLM scores each answer on two metrics:
#    - **Faithfulness** — is the answer grounded in the retrieved pages, or did the LLM hallucinate?
#    - **Correctness** — does the answer match the reference answer from step 2?

# > Scores will likely be slightly optimistic because the generated questions are
# > simpler than real user questions. Use this as a sanity check, not a hard benchmark.
# """)

    if not groq_api_key:
        st.error("GROQ_API_KEY required to run evaluation.")
    else:
        col_a, col_b = st.columns([1, 2])
        with col_a:
            n_questions = st.slider(
                "Number of questions", min_value=3, max_value=20, value=5, step=1
            )
            seed = st.number_input(
                "Random seed (for reproducibility)", value=42, step=1
            )
        with col_b:
            st.info(
                f"This will make roughly **{n_questions * 3}** Groq API calls "
                f"(generate + answer + judge per question).  "
                f"Estimated time: **{n_questions * 15}–{n_questions * 25} seconds**."
            )

        if st.button("▶ Run Evaluation", type="primary"):

            

            def _generate_qa(image_path: str) -> dict:
                """Ask the vision LLM to write one Q&A pair grounded in this page."""
                b64 = encode_image_base64(image_path)
                prompt = (
                    "You are building a test dataset to evaluate a RAG system.\n"
                    "Look at this document page carefully.\n"
                    "Write ONE specific factual question whose answer appears ONLY on this page "
                    "(not general knowledge). Also write the correct short answer.\n\n"
                    "Rules:\n"
                    "- The question must be answerable from this page alone.\n"
                    "- Prefer questions about specific numbers, names, methods, or findings.\n"
                    "- Keep the answer under 40 words.\n\n"
                    "Respond ONLY in this JSON format (no markdown, no extra text):\n"
                    '{"question": "...", "answer": "..."}'
                )
                resp = groq_client.chat.completions.create(
                    model=VISION_MODEL,
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ]}],
                    temperature=0.3,
                    max_tokens=200,
                )
                return parse_json_response(resp.choices[0].message.content)


            def _query_rag(question: str) -> tuple[str, list[dict]]:
                """Run a question through the same RAG pipeline as the Query tab."""
                res = collection.query(
                    query_texts=[question],
                    n_results=5,
                    include=["uris", "metadatas"],
                )
                uris      = res["uris"][0]
                metadatas = res["metadatas"][0]

                prompt_content = [{
                    "type": "text",
                    "text": (
                        f"Answer this question based ONLY on the provided document pages.\n"
                        f"Be concise and factual. If the answer is not visible, say so.\n\n"
                        f"Question: {question}"
                    ),
                }]
                for uri in uris:
                    path = resolve_image_path(uri)
                    if os.path.exists(path):
                        prompt_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encode_image_base64(path)}"
                            },
                        })

                resp = groq_client.chat.completions.create(
                    model=VISION_MODEL,
                    messages=[{"role": "user", "content": prompt_content}],
                    temperature=0.1,
                    max_tokens=300,
                )
                return resp.choices[0].message.content.strip(), metadatas


            def _judge(
                question: str,
                reference: str,
                rag_answer: str,
                retrieved_meta: list[dict],
            ) -> dict:
                """Score the RAG answer using a text-only judge LLM (faster, free)."""
                context_str = "; ".join(
                    f"{m.get('source_paper', '?')} p.{m.get('page_number', '?')}"
                    for m in retrieved_meta
                )
                prompt = f"""You are evaluating a RAG system. Score the following on two criteria.

Question: {question}
Reference Answer (ground truth): {reference}
RAG System Answer: {rag_answer}
Retrieved from: {context_str}

Score each criterion from 1 to 5:
- faithfulness: Does the RAG answer only state things grounded in its source context? (1=hallucinated, 5=fully grounded)
- correctness: How closely does the RAG answer match the reference answer? (1=completely wrong, 5=fully correct)

Respond ONLY in this JSON format (no markdown, no extra text):
{{"faithfulness": <1-5>, "correctness": <1-5>, "reasoning": "<one sentence>"}}"""

                resp = groq_client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=150,
                )
                return parse_json_response(resp.choices[0].message.content)


            
            all_items = collection.get(include=["uris", "metadatas"])
            all_pages = list(zip(
                all_items["ids"],
                all_items["uris"],
                all_items["metadatas"],
            ))
            random.seed(int(seed))
            sampled = random.sample(all_pages, min(n_questions, len(all_pages)))

            
            progress_bar = st.progress(0, text="Starting evaluation…")
            status_box   = st.empty()

            eval_results       = []
            faithfulness_scores = []
            correctness_scores  = []
            skipped = 0

            for i, (doc_id, uri, meta) in enumerate(sampled):
                paper    = meta.get("source_paper", "unknown")
                page     = meta.get("page_number", "?")
                img_path = resolve_image_path(uri)

                progress_bar.progress(
                    i / len(sampled),
                    text=f"Question {i+1}/{len(sampled)} — {paper}, page {page}",
                )

                
                try:
                    status_box.info(
                        f"**[{i+1}/{len(sampled)}]** Generating question from "
                        f"*{paper}* — page {page}…"
                    )
                    qa        = _generate_qa(img_path)
                    question  = qa["question"]
                    reference = qa["answer"]
                except Exception as e:
                    status_box.warning(f"QA generation failed ({paper} p.{page}): {e}")
                    skipped += 1
                    continue

                
                try:
                    status_box.info(
                        f"**[{i+1}/{len(sampled)}]** Querying RAG: "
                        f"*{question[:70]}{'…' if len(question) > 70 else ''}*"
                    )
                    rag_answer, retrieved_meta = _query_rag(question)
                except Exception as e:
                    status_box.warning(f"RAG query failed: {e}")
                    skipped += 1
                    continue

                
                try:
                    status_box.info(
                        f"**[{i+1}/{len(sampled)}]** Judging answer…"
                    )
                    scores = _judge(question, reference, rag_answer, retrieved_meta)
                    faithfulness_scores.append(scores["faithfulness"])
                    correctness_scores.append(scores["correctness"])
                except Exception as e:
                    status_box.warning(f"Judging failed: {e}")
                    skipped += 1
                    continue

                eval_results.append({
                    "source_paper"   : paper,
                    "page"           : page,
                    "question"       : question,
                    "reference"      : reference,
                    "rag_answer"     : rag_answer,
                    "retrieved_pages": [
                        f"{m.get('source_paper')} p.{m.get('page_number')}"
                        for m in retrieved_meta
                    ],
                    "faithfulness"   : scores["faithfulness"],
                    "correctness"    : scores["correctness"],
                    "reasoning"      : scores["reasoning"],
                })

            progress_bar.progress(1.0, text="Evaluation complete!")
            status_box.empty()

            
            n = len(eval_results)

            if n == 0:
                st.error(
                    "All evaluation steps failed. "
                    "Check your Groq API key and ChromaDB connection."
                )
            else:
                avg_f = sum(faithfulness_scores) / n
                avg_c = sum(correctness_scores)  / n

                st.success(f" Evaluated {n} question(s) — {skipped} skipped due to errors")

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Questions Evaluated", n)
                mc2.metric("Avg Faithfulness",    f"{avg_f:.2f} / 5")
                mc3.metric("Avg Correctness",     f"{avg_c:.2f} / 5")

                
                st.subheader("Results")
                import pandas as pd
                df = pd.DataFrame([{
                    "Paper"           : r["source_paper"],
                    "Page"            : r["page"],
                    "Question"        : (r["question"][:80] + "…"
                                         if len(r["question"]) > 80
                                         else r["question"]),
                    "Faithfulness ↑"  : r["faithfulness"],
                    "Correctness ↑"   : r["correctness"],
                    "Judge Reasoning" : r["reasoning"],
                } for r in eval_results])
                st.dataframe(df, use_container_width=True)

                
                with st.expander("View full Q&A details"):
                    for r in eval_results:
                        st.markdown(f"**Question:** {r['question']}")
                        st.markdown(f"**Reference answer:** {r['reference']}")
                        st.markdown(f"**RAG answer:** {r['rag_answer']}")
                        st.markdown(
                            f"**Retrieved pages:** {', '.join(r['retrieved_pages'])}"
                        )
                        st.markdown(
                            f"**Scores:** Faithfulness {r['faithfulness']}/5 "
                            f"| Correctness {r['correctness']}/5"
                        )
                        st.divider()

                
                output_json = json.dumps({
                    "summary": {
                        "total_evaluated" : n,
                        "skipped"         : skipped,
                        "avg_faithfulness": round(avg_f, 2),
                        "avg_correctness" : round(avg_c, 2),
                    },
                    "results": eval_results,
                }, indent=2)

                st.download_button(
                    label="⬇ Download full results (JSON)",
                    data=output_json,
                    file_name="eval_results.json",
                    mime="application/json",
                )
