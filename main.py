import os
import json
import time
from datetime import datetime, timezone
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_community.tools.tavily_search import TavilySearchResults
from openai import OpenAI

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("OPENAI_API_KEY environment variable is required")

import httpx
http_client = httpx.Client()
client = OpenAI(http_client=http_client)

class State(TypedDict, total=False):
    job_title: str
    location: str
    experience: str
    search_results: dict
    job_json: dict  
graph_builder = StateGraph(State)

def _ensure_env():
    missing = []
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.getenv("TAVILY_API_KEY"):
        missing.append("TAVILY_API_KEY")
    if missing:
        raise EnvironmentError(f"Missing required env vars: {', '.join(missing)}")

def _format_search_results(items) -> str:
    """
    TavilySearchResults returns list[dict] with keys like 'content', 'url'.
    Convert to clean bullet points for prompting.
    """
    if not items:
        return "- No reliable public info found."
    lines = []
    for it in items:
        content = (it.get("content") or it.get("snippet") or "").strip()
        url = it.get("url") or it.get("source") or ""
        if content and url:
            lines.append(f"- {content} ({url})")
        elif content:
            lines.append(f"- {content}")
        elif url:
            lines.append(f"- {url}")
    return "\n".join(lines) if lines else "- No reliable public info found."

def _safe_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)

def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _dedupe_str_list(seq):
    seen, out = set(), []
    for s in seq or []:
        if not isinstance(s, str):
            continue
        key = s.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(s.strip())
    return out

def _strict_json_loads(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start : end + 1])
    except Exception:
        pass
    raise ValueError("Model did not return valid JSON")

def input_node(state: State):
    """
    Validate minimal inputs & env.
    """
    start_time = time.time()
    print("🔍 Validating inputs and environment...")
    
    if not state.get("job_title") or not state.get("location") or not state.get("experience"):
        raise ValueError("job_title, location, and experience are required.")
    _ensure_env()
    
    elapsed = time.time() - start_time
    print(f"   ✅ Input validation completed in {elapsed:.3f} seconds")
    return state

graph_builder.add_node("input", input_node)

tavily = TavilySearchResults(max_results=5)

def search_node(state: State):
    """
    Multi-pronged search tailored by title, location, and experience.
    We gather enough context so the model can infer
    industry, Education, company_name, location_type, and skills.
    """
    start_time = time.time()
    print("🔍 Starting market research with Tavily...")
    
    title = state["job_title"]
    location = state["location"]
    exp = state["experience"]

    def _try(q):
        try:
            return tavily.invoke(q)
        except Exception:
            return []

    search_queries = [
        ("role_by_location", f"Job description and responsibilities of a {title} in {location}"),
        ("required_qualifications", f"Required qualifications for {title} roles in {location}"),
        ("experience_expectations", f"{exp} {title} responsibilities and qualifications"),
        ("benefits_norms", f"Common benefits for {title} roles in {location}"),
        ("market_snapshot", f"{title} hiring trends and industry in {location}"),
        ("sample_postings", f"{title} openings in {location} job posting responsibilities qualifications"),
        ("education_requirements", f"Typical education requirements for {title} in {location}"),
    ]
    
    results = {}
    for key, query in search_queries:
        query_start = time.time()
        results[key] = _try(query)
        query_time = time.time() - query_start
        print(f"   📊 {key}: {query_time:.2f}s")

    elapsed = time.time() - start_time
    print(f"   ✅ Market research completed in {elapsed:.2f} seconds")
    return {"search_results": results}

graph_builder.add_node("search", search_node)

def job_json_node(state: State):
    """
    Produce the strict JSON matching the requested schema.
    All fields other than the 3 inputs are inferred from web bullets.
    """
    start_time = time.time()
    print("🤖 Generating job description with OpenAI...")
    
    title = state["job_title"]
    location = state["location"]
    exp = state["experience"]
    sr = state.get("search_results", {})

    format_start = time.time()
    role_loc = _format_search_results(sr.get("role_by_location"))
    req_quals = _format_search_results(sr.get("required_qualifications"))
    exp_info = _format_search_results(sr.get("experience_expectations"))
    benefits = _format_search_results(sr.get("benefits_norms"))
    market = _format_search_results(sr.get("market_snapshot"))
    postings = _format_search_results(sr.get("sample_postings"))
    edu = _format_search_results(sr.get("education_requirements"))
    format_time = time.time() - format_start
    print(f"   📝 Search results formatting: {format_time:.3f}s")

    prompt = f"""
You are a precise HR/recruiting writer. Return a SINGLE valid JSON object ONLY,
matching this EXACT schema (keys and casing EXACT):

{{
  "timestamp": "",
  "params": {{
    "job_title": "",
    "industry": "",
    "Education": "",
    "company_name": "",
    "location_type": "",
    "experience": "",
    "required_skills": [],
    "preferred_skills": []
  }},
  "outputs": {{
    "sections": {{
      "Executive Summary": "",
      "Key Responsibilities": [],
      "Required Qualifications": [],
      "Preferred Qualifications": [],
      "What We Offer": [],
      "skills": []
    }}
  }}
}}

Inputs (MUST be used verbatim in 'params'):
- job_title: {title}
- location: {location}   # Use this to tailor content, but it is NOT a key in params
- experience: {exp}

Inference rules:
- Infer remaining 'params' fields from the research bullets below.
  * "industry": best-fit sector in this market (e.g., "Software & Technology" if unclear).
  * "Education": typical minimum degree for this role in this location; include field if common (e.g., "Bachelor’s in Computer Science or related").
  * "company_name": only if a single company is strongly implied by research; otherwise leave "".
  * "location_type": infer common working model in this market (e.g., "Onsite", "Hybrid", "Remote"); if unclear, choose the most common locally and state it plainly.
  * "required_skills" and "preferred_skills": infer from postings and norms; keep bullets concise; avoid duplicates.
- Populate 'outputs.sections' using the research; do NOT invent exact salary numbers. Use generic phrases for comp/benefits if specifics are not present.
- Keep "Executive Summary" to 2–4 crisp sentences tailored to the role, location, and experience.
- Output VALID JSON only (no markdown, no explanations).

Research bullets:
- Role by location:
{role_loc}

- Required qualifications:
{req_quals}

- Expectations by experience:
{exp_info}

- Benefits norms:
{benefits}

- Market snapshot:
{market}

- Sample postings:
{postings}

- Education requirements:
{edu}
""".strip()

    api_start = time.time()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    api_time = time.time() - api_start
    print(f"   🤖 OpenAI API call: {api_time:.3f}s")
    
    parse_start = time.time()
    raw = resp.choices[0].message.content or "{}"
    obj = _strict_json_loads(raw)
    parse_time = time.time() - parse_start
    print(f"   📋 JSON parsing: {parse_time:.3f}s")

    obj["timestamp"] = _now_iso_utc()
    obj.setdefault("params", {})
    obj["params"].setdefault("job_title", title)
    obj["params"].setdefault("experience", exp)
    for k in ["industry", "Education", "company_name", "location_type", "required_skills", "preferred_skills"]:
        obj["params"].setdefault(k, [] if "skills" in k else "")

    sections = obj.setdefault("outputs", {}).setdefault("sections", {})
    for k in ["Executive Summary", "Key Responsibilities", "Required Qualifications", "Preferred Qualifications", "What We Offer", "skills"]:
        if k == "Executive Summary":
            sections.setdefault(k, "")
        else:
            sections.setdefault(k, [])

    merged_skills = _dedupe_str_list(
        (obj["params"].get("required_skills") or [])
        + (obj["params"].get("preferred_skills") or [])
        + (sections.get("skills") or [])
    )
    sections["skills"] = merged_skills

    elapsed = time.time() - start_time
    print(f"   ✅ Job description generation completed in {elapsed:.2f} seconds")
    return {"job_json": obj}

graph_builder.add_node("generate_job_json", job_json_node)
graph_builder.add_edge(START, "input")
graph_builder.add_edge("input", "search")
graph_builder.add_edge("search", "generate_job_json")
graph_builder.add_edge("generate_job_json", END)

graph = graph_builder.compile()

def run_graph(
    job_title: str,
    location: str,
    experience: str,
    outdir: str = "."
) -> dict:
    """
    Execute the workflow using ONLY the three inputs.
    The rest of the fields are inferred from Tavily search results.
    Writes {job_title}_{location}_{experience}.json (sanitized) and returns the dict.
    """
    start_time = time.time()
    print(f"🚀 Starting job description generation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📋 Job Title: {job_title}")
    print(f"📍 Location: {location}")
    print(f"⏱️  Experience: {experience}")
    print("-" * 60)
    
    state: State = {
        "job_title": job_title,
        "location": location,
        "experience": experience,
    }

    graph_start = time.time()
    print("🔍 Starting research and generation process...")
    result = graph.invoke(state)
    graph_end = time.time()
    
    obj = result["job_json"]

    write_start = time.time()
    fname = f"{_safe_filename(job_title)}_{_safe_filename(location)}_{_safe_filename(experience)}.json"
    fpath = os.path.join(outdir, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    write_end = time.time()

    total_time = time.time() - start_time
    graph_time = graph_end - graph_start
    write_time = write_end - write_start
    
    print("-" * 60)
    print("⏱️  TIMING RESULTS:")
    print(f"   📊 Graph execution time: {graph_time:.2f} seconds")
    print(f"   💾 File writing time: {write_time:.3f} seconds")
    print(f"   🎯 Total execution time: {total_time:.2f} seconds")
    print(f"✅ Successfully wrote {fpath}")
    print(f"🏁 Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return obj

if __name__ == "__main__":
    run_graph(
        job_title="Software Engineer",
        location="Dhaka, Bangladesh",
        experience="3–5 years",
    )
