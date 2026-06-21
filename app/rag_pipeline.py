"""
rag_pipeline.py
Core RAG logic + Fit Score + Skill Gap Coaching + Resume Tailoring, powered by Groq LLM.
"""

import os
from groq import Groq
from app.embeddings import search_similar_chunks

# Groq client setup - works both locally (.env) and on Streamlit Cloud (secrets)
try:
    import streamlit as st
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")

client = Groq(api_key=api_key)

GROQ_MODEL = "openai/gpt-oss-20b"


def calculate_fit_score(resume_text, jd_text):
    fit_prompt = (
        "You are a recruitment analyst. Compare the following RESUME against the JOB DESCRIPTION "
        "and assess how well the candidate fits this role.\n\n"
        "RESUME:\n" + resume_text + "\n\n"
        "JOB DESCRIPTION:\n" + jd_text + "\n\n"
        "Provide your response in this exact format:\n\n"
        "Fit Score: [a number from 0-100]%\n\n"
        "Strengths:\n"
        "- [2-3 bullet points on what matches well]\n\n"
        "Gaps:\n"
        "- [2-3 bullet points on what's missing or weak]\n\n"
        "Verdict: [One sentence overall recommendation]"
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a precise, honest recruitment analyst. Be specific and avoid generic statements."},
            {"role": "user", "content": fit_prompt}
        ],
        temperature=0.3,
        max_tokens=600,
    )

    return response.choices[0].message.content.strip()


def tailor_resume(resume_text, jd_text, fit_analysis):
    tailor_prompt = (
        "You are an expert ATS resume writer. Rewrite the RESUME below into a complete, "
        "polished, ATS-friendly resume tailored specifically for the JOB DESCRIPTION provided. "
        "Use the FIT ANALYSIS to know what to emphasize and what gaps to handle gracefully.\n\n"
        "ORIGINAL RESUME:\n" + resume_text + "\n\n"
        "JOB DESCRIPTION:\n" + jd_text + "\n\n"
        "FIT ANALYSIS:\n" + fit_analysis + "\n\n"
        "Rules:\n"
        "- Output a COMPLETE, ready-to-use resume, not suggestions or comparisons.\n"
        "- Use these exact section headings, in this order, only if the original resume has "
        "relevant content for them (skip empty sections):\n"
        "  ## Professional Summary\n"
        "  ## Skills\n"
        "  ## Experience\n"
        "  ## Projects\n"
        "  ## Education\n"
        "  ## Certifications\n"
        "- Professional Summary: 2-3 lines, written using strong keywords directly from the "
        "job description, truthfully reflecting the candidate's real background.\n"
        "- Skills: list skills from the original resume, reordered/regrouped to surface the ones "
        "most relevant to this JD first. You may rename a skill to match the JD's exact wording "
        "ONLY if it is genuinely the same skill (e.g. 'Generative AI' to match JD phrasing for "
        "the candidate's existing LLM/RAG project work). Never add a skill the candidate doesn't have.\n"
        "- Experience and Projects: rewrite each bullet point using strong action verbs and JD "
        "keywords where truthfully applicable, keeping all facts, numbers, and outcomes from the "
        "original. Do not invent metrics, responsibilities, or outcomes not present in the original.\n"
        "- Education and Certifications: keep exactly as in the original, just reformatted cleanly.\n"
        "- Formatting: use '## ' for section headings and '- ' for bullet points. No tables, "
        "no colors, no graphics - this must be ATS-parseable plain structured text.\n"
        "- Do not include any preamble, explanation, or commentary - output ONLY the resume content."
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a precise ATS resume writer. Never fabricate experience, skills, or metrics. Output only the resume itself, no commentary."},
            {"role": "user", "content": tailor_prompt}
        ],
        temperature=0.3,
        max_tokens=1400,
    )

    return response.choices[0].message.content.strip()


def generate_cover_letter(resume_text, jd_text, fit_analysis):
    cover_letter_prompt = (
        "You are an expert cover letter writer. Write a complete, compelling cover letter for this "
        "candidate applying to this job, based on their RESUME, the JOB DESCRIPTION, and the FIT ANALYSIS.\n\n"
        "RESUME:\n" + resume_text + "\n\n"
        "JOB DESCRIPTION:\n" + jd_text + "\n\n"
        "FIT ANALYSIS:\n" + fit_analysis + "\n\n"
        "Rules:\n"
        "- Write a complete cover letter, 3-4 paragraphs, ready to send.\n"
        "- Paragraph 1: A strong opening stating the role and one genuine hook (a real strength or "
        "achievement from the resume that's directly relevant to this JD).\n"
        "- Paragraph 2-3: Connect 2-3 specific real experiences/projects from the resume to the job's "
        "actual requirements. Use real details (project names, technologies, outcomes) - never invent.\n"
        "- Final paragraph: A confident, brief closing expressing interest and inviting next steps.\n"
        "- Tone: professional but warm, not robotic or generic. Avoid cliches like 'I am writing to "
        "express my interest' - start with something more specific and engaging.\n"
        "- Do not invent any experience, company names, or facts not present in the resume.\n"
        "- Do not include a header with date/address placeholders - just the letter body starting with "
        "'Dear Hiring Manager,' and ending with a sign-off and the candidate's name from the resume."
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a skilled cover letter writer. Be specific, genuine, and never fabricate experience."},
            {"role": "user", "content": cover_letter_prompt}
        ],
        temperature=0.5,
        max_tokens=700,
    )

    return response.choices[0].message.content.strip()


def generate_interview_prep(resume_text, jd_text, fit_analysis):
    interview_prompt = (
        "You are an experienced interview coach. Based on this candidate's RESUME, the JOB DESCRIPTION, "
        "and their FIT ANALYSIS, prepare them for an interview for this role.\n\n"
        "RESUME:\n" + resume_text + "\n\n"
        "JOB DESCRIPTION:\n" + jd_text + "\n\n"
        "FIT ANALYSIS:\n" + fit_analysis + "\n\n"
        "Provide your response in this format:\n\n"
        "## Likely Technical Questions\n"
        "List 4-5 technical questions specifically relevant to this JD's requirements and this "
        "candidate's background. For each, give a one-line pointer on how to approach answering it "
        "using the candidate's real experience.\n\n"
        "## Likely Behavioral Questions\n"
        "List 3-4 behavioral questions likely for this type of role. For each, suggest a real "
        "project/experience from the resume that could anchor a STAR-format answer.\n\n"
        "## Questions About Your Gaps\n"
        "Based on the fit analysis gaps, list 1-2 questions the interviewer might ask to probe those "
        "gaps, and a brief honest way to address them confidently.\n\n"
        "## Questions to Ask Them\n"
        "Suggest 2-3 thoughtful questions the candidate could ask the interviewer about this specific role.\n\n"
        "Keep it practical and specific to this candidate and this job, not generic interview advice.\n\n"
        "Formatting rules: use '## ' for section headings and '- ' for bullet points only. "
        "Do NOT use markdown tables, pipe characters (|), or any table formatting. "
        "Each question and its guidance should be its own bullet point in plain text."
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a practical interview coach. Be specific to the candidate and role, not generic."},
            {"role": "user", "content": interview_prompt}
        ],
        temperature=0.4,
        max_tokens=1800,
    )

    return response.choices[0].message.content.strip()


def chat_response(user_message, faiss_index, chunks, jd_text, resume_text, fit_analysis, chat_history):
    """
    Single entry point for the chatbot. Decides what context to use based on
    what's available (JD only, JD+resume, JD+resume+fit), and answers naturally
    without the user needing to follow any steps.

    Returns (answer_text, result_type) where result_type is one of:
    "none", "tailoring", "cover_letter", "interview_prep"
    """
    has_jd = jd_text is not None and len(jd_text.strip()) > 0
    has_resume = resume_text is not None and len(resume_text.strip()) > 0
    has_fit = fit_analysis is not None and len(fit_analysis.strip()) > 0

    if not has_jd:
        return ("I don't have a job description yet. Upload one using the JD attach button "
                "below, or paste it directly into the chat and I'll take it from there."), "none"

    affirmative_replies = ["yes", "yes please", "sure", "go ahead", "ok", "okay", "yep", "please do", "yes please do"]
    last_bot_message = ""
    if chat_history:
        for msg in reversed(chat_history):
            if msg["role"] == "assistant":
                last_bot_message = msg["content"].lower()
                break

    user_said_yes = user_message.strip().lower() in affirmative_replies

    wants_tailoring = (
        (user_said_yes and "tailor" in last_bot_message) or
        "tailor" in user_message.lower() or
        "rewrite my resume" in user_message.lower()
    )
    wants_cover_letter = (
        (user_said_yes and "cover letter" in last_bot_message) or
        "cover letter" in user_message.lower()
    )
    wants_interview_prep = (
        (user_said_yes and "interview" in last_bot_message) or
        "interview" in user_message.lower() and (
            "prep" in user_message.lower() or "question" in user_message.lower() or "ready" in user_message.lower()
        )
    )

    needs_resume_and_fit_message = (
        "I'd need both your resume and a calculated fit score first. "
        "Attach your resume if you haven't already, and I'll take it from there."
    )

    if wants_tailoring:
        if has_resume and has_fit:
            return tailor_resume(resume_text, jd_text, fit_analysis), "tailoring"
        return needs_resume_and_fit_message, "none"

    if wants_cover_letter:
        if has_resume and has_fit:
            return generate_cover_letter(resume_text, jd_text, fit_analysis), "cover_letter"
        return needs_resume_and_fit_message, "none"

    if wants_interview_prep:
        if has_resume and has_fit:
            return generate_interview_prep(resume_text, jd_text, fit_analysis), "interview_prep"
        return needs_resume_and_fit_message, "none"

    relevant_chunks = []
    if faiss_index is not None and chunks:
        relevant_chunks = search_similar_chunks(query=user_message, faiss_index=faiss_index, chunks=chunks, top_k=3)
    jd_context = "\n\n---\n\n".join(relevant_chunks) if relevant_chunks else jd_text[:2000]

    context_prompt = (
        "You are JobSense, a friendly conversational assistant (like a helpful colleague) that helps "
        "a candidate understand a job description, their fit for it, how to close skill gaps, tailor "
        "their resume, write a cover letter, and prepare for the interview for this job. "
        "Never mention 'steps' or tell the user to follow a process - just respond naturally to what they ask.\n\n"
        "RELEVANT JOB DESCRIPTION CONTEXT:\n" + jd_context + "\n\n"
    )

    if has_fit:
        context_prompt += (
            "FIT ANALYSIS ALREADY COMPUTED FOR THIS CANDIDATE:\n" + fit_analysis + "\n\n"
            "Depending on context, you can offer to: tailor their resume, write a cover letter, or "
            "help them prep for the interview for this role - mention these naturally when relevant, "
            "not all at once.\n\n"
        )
    elif has_resume:
        context_prompt += (
            "CANDIDATE RESUME (fit score not yet calculated):\n" + resume_text[:1500] + "\n\n"
            "If the user asks about their fit, gaps, or chances, let them know you can calculate "
            "an exact fit score now that you have both documents - just ask them to say 'check my fit' "
            "or similar, OR calculate it yourself conversationally using the resume and JD above.\n\n"
        )
    else:
        context_prompt += (
            "No resume has been provided yet. If the user asks about their personal fit, skill gaps, "
            "resume tailoring, a cover letter, or interview prep, ask them to upload or paste their "
            "resume using the resume attach button so you can compare it against this job.\n\n"
        )

    context_prompt += (
        "Answer the user's message below conversationally and specifically. "
        "Do not invent information not present above. "
        "Never use markdown tables or pipe characters (|) - use plain text or '- ' bullet points instead. "
        "If your answer would naturally be long (e.g. multiple questions, a list of examples, or a "
        "detailed explanation), make sure to complete every item fully - never stop mid-sentence or "
        "mid-list. It is fine to be thorough."
    )

    messages = [{"role": "system", "content": context_prompt}]

    if chat_history:
        for msg in chat_history[-8:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_message})

    # Detect signals that this question likely needs a longer, detailed answer
    long_answer_signals = [
        "questions", "list", "examples", "in detail", "explain", "describe",
        "give me", "interview", "prepare", "scenario", "situation", "all the",
        "write", "draft", "summary", "summarize"
    ]
    user_lower = user_message.lower()
    needs_longer_answer = any(signal in user_lower for signal in long_answer_signals)
    dynamic_max_tokens = 1200 if needs_longer_answer else 500

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=dynamic_max_tokens,
    )

    return response.choices[0].message.content.strip(), "none"
