"""
rag_pipeline.py
Core RAG logic + Fit Score + Resume Tailoring + Cover Letter + Interview Prep,
powered by Groq LLM. General chat responses stream live, like ChatGPT.
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
    # Hard safety check: refuse to proceed if resume content looks insufficient,
    # rather than letting the LLM fill gaps with invented experience.
    if not resume_text or len(resume_text.strip()) < 100:
        return (
            "I can't generate a tailored resume right now - the resume text I have is too short "
            "or empty, which usually means the PDF didn't extract properly. Could you re-attach "
            "your resume, or paste its text directly into the chat?"
        )

    tailor_prompt = (
        "You are an expert ATS resume writer. Rewrite the RESUME below into a complete, "
        "polished, ATS-friendly resume tailored specifically for the JOB DESCRIPTION provided. "
        "Use the FIT ANALYSIS to know what to emphasize and what gaps to handle gracefully.\n\n"
        "ORIGINAL RESUME (this is the ONLY source of the candidate's real experience):\n"
        "-----\n" + resume_text + "\n-----\n\n"
        "JOB DESCRIPTION:\n" + jd_text + "\n\n"
        "FIT ANALYSIS:\n" + fit_analysis + "\n\n"
        "CRITICAL RULE - READ CAREFULLY:\n"
        "Every single company name, job title, date, project name, skill, and metric in your output "
        "MUST come from the ORIGINAL RESUME text above. If the original resume does not mention a "
        "specific company, job title, or project, you must NOT invent one. If the original resume's "
        "experience section is thin, missing, or unclear, keep your output thin too - reflect reality, "
        "do not pad it with fabricated jobs or projects. If you are about to write a company name, "
        "job title, or achievement that you cannot point to directly in the ORIGINAL RESUME text "
        "above, delete it and do not include it. A resume with less content but 100% truthful is "
        "far better than a padded resume with invented experience.\n\n"
        "Rules:\n"
        "- Output a COMPLETE, ready-to-use resume, not suggestions or comparisons.\n"
        "- Use these exact section headings, in this order, only if the original resume has "
        "relevant content for them (skip sections with no real content - do not invent content to fill them):\n"
        "  ## Professional Summary\n"
        "  ## Skills\n"
        "  ## Experience\n"
        "  ## Projects\n"
        "  ## Education\n"
        "  ## Certifications\n"
        "- Professional Summary: 2-3 lines, written using strong keywords directly from the "
        "job description, truthfully reflecting ONLY the candidate's real background as written "
        "in the original resume.\n"
        "- Skills: list skills from the original resume only, reordered/regrouped to surface the ones "
        "most relevant to this JD first. Never add a skill the candidate doesn't have.\n"
        "- Experience and Projects: rewrite each bullet point that EXISTS in the original using "
        "stronger action verbs and JD keywords where truthfully applicable, keeping all facts, "
        "numbers, and outcomes from the original exactly as given. Do not invent metrics, "
        "responsibilities, companies, or outcomes not present in the original. If the candidate has "
        "no formal work experience, OMIT the Experience section entirely rather than inventing one.\n"
        "- Education and Certifications: keep exactly as in the original, just reformatted cleanly.\n"
        "- Formatting: use '## ' for section headings and '- ' for bullet points. No tables, "
        "no colors, no graphics - this must be ATS-parseable plain structured text.\n"
        "- Do not include any preamble, explanation, or commentary - output ONLY the resume content."
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": (
                "You are a precise ATS resume writer. You NEVER invent companies, job titles, dates, "
                "projects, skills, or metrics that are not explicitly present in the candidate's original "
                "resume text provided to you. If the original resume is thin or has no work experience, "
                "your output stays thin and honest rather than padded with fabricated content. Fabricating "
                "experience is a serious failure - the candidate could be asked about it in a real interview "
                "and caught lying. Output only the resume itself, no commentary."
            )},
            {"role": "user", "content": tailor_prompt}
        ],
        temperature=0.1,
        max_tokens=2500,
    )

    result = response.choices[0].message.content.strip()
    was_truncated = response.choices[0].finish_reason == "length"

    # Safety check: retry if the model either got cut off due to length, or
    # didn't follow the '## Heading' format we need for proper docx rendering.
    needs_retry = was_truncated or "## " not in result

    if needs_retry:
        retry_prompt = tailor_prompt + (
            "\n\nIMPORTANT: Be concise and efficient with wording so the FULL resume fits completely - "
            "every section must be fully completed, never cut off partway through. Use '## ' for each "
            "section heading on its own line, exactly as instructed above."
        )
        retry_response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": (
                    "You are a precise ATS resume writer. You NEVER invent companies, job titles, dates, "
                    "projects, skills, or metrics not in the original resume. You ALWAYS use '## Heading' "
                    "format for sections, and you ALWAYS complete every section fully - never stop mid-list "
                    "or mid-sentence. Output only the resume itself, no commentary."
                )},
                {"role": "user", "content": retry_prompt}
            ],
            temperature=0.1,
            max_tokens=3000,
        )
        retry_result = retry_response.choices[0].message.content.strip()
        retry_truncated = retry_response.choices[0].finish_reason == "length"

        # Use the retry only if it's actually better (not also truncated, or longer than original)
        if not retry_truncated or len(retry_result) > len(result):
            result = retry_result

    warning = _check_for_likely_fabrication(result, resume_text)
    if warning:
        result = warning + "\n\n" + result

    return result


def _check_for_likely_fabrication(tailored_text, original_resume_text):
    """
    Lightweight heuristic check: looks for an 'Experience' section in the
    tailored output that wasn't present at all in the original resume, which
    strongly suggests fabricated work history. Returns a warning string if
    suspicious, or None if it looks fine.
    """
    original_lower = original_resume_text.lower()
    tailored_lower = tailored_text.lower()

    original_has_experience_signals = any(
        signal in original_lower for signal in
        ["intern", "experience", "worked at", "employed", "company", "present)", "- present"]
    )

    tailored_has_experience_section = "## experience" in tailored_lower

    if tailored_has_experience_section and not original_has_experience_signals:
        return (
            "⚠️ Heads up: this tailored resume includes an Experience section, but I couldn't "
            "find clear work experience in your original resume. Please double-check this section "
            "carefully before using it - if anything looks unfamiliar, let me know and I'll regenerate "
            "it without that section."
        )
    return None


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


def detect_intent(user_message, chat_history):
    """
    Figures out what the user wants based on their message and recent chat history.
    Returns one of: "tailoring", "cover_letter", "interview_prep", "general"
    """
    affirmative_replies = ["yes", "yes please", "sure", "go ahead", "ok", "okay", "yep", "please do", "yes please do"]
    last_bot_message = ""
    if chat_history:
        for msg in reversed(chat_history):
            if msg["role"] == "assistant":
                last_bot_message = msg["content"].lower()
                break

    user_said_yes = user_message.strip().lower() in affirmative_replies

    if (user_said_yes and "tailor" in last_bot_message) or "tailor" in user_message.lower() or "rewrite my resume" in user_message.lower():
        return "tailoring"
    if (user_said_yes and "cover letter" in last_bot_message) or "cover letter" in user_message.lower():
        return "cover_letter"
    if (user_said_yes and "interview" in last_bot_message) or (
        "interview" in user_message.lower() and (
            "prep" in user_message.lower() or "question" in user_message.lower() or "ready" in user_message.lower()
        )
    ):
        return "interview_prep"
    return "general"


def build_general_chat_messages(user_message, faiss_index, chunks, jd_text, resume_text, fit_analysis, chat_history):
    """Builds the messages list for a general conversational response."""
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

    has_fit = fit_analysis is not None and len(fit_analysis.strip()) > 0
    has_resume = resume_text is not None and len(resume_text.strip()) > 0

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

    user_lower_check = user_message.strip().lower()
    if user_lower_check in ["continue", "continue please", "keep going", "go on"]:
        context_prompt += (
            "\n\nThe user is asking you to CONTINUE your previous response, which got cut off "
            "mid-way. Look at your last message in the conversation history below, and continue "
            "exactly where it left off - do not repeat content you already covered, do not restart "
            "the explanation, just pick up the next point naturally."
        )

    messages = [{"role": "system", "content": context_prompt}]

    if chat_history:
        for msg in chat_history[-8:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_message})
    return messages


def stream_chat_response(user_message, faiss_index, chunks, jd_text, resume_text, fit_analysis, chat_history):
    """
    Generator version of the general chat response - yields text chunks as they
    arrive from Groq, enabling a live 'typing' effect in the UI.
    Only used for the general conversational path (not tailoring/cover letter/
    interview prep, which return complete structured documents at once).
    """
    has_jd = jd_text is not None and len(jd_text.strip()) > 0
    if not has_jd:
        yield ("I don't have a job description yet. Upload one using the JD attach button "
               "below, or paste it directly into the chat and I'll take it from there.")
        return

    user_lower = user_message.lower()

    very_long_signals = [
        "roadmap", "plan", "comprehensive", "deep dive", "deepen", "everything about",
        "full guide", "complete guide", "all topics", "cover everything", "study plan",
        "preparation plan", "learning path"
    ]
    long_answer_signals = [
        "questions", "list", "examples", "in detail", "explain", "describe",
        "give me", "interview", "prepare", "scenario", "situation", "all the",
        "write", "draft", "summary", "summarize", "topics"
    ]

    if user_lower.strip() in ["continue", "continue please", "keep going", "go on"]:
        dynamic_max_tokens = 3500
    elif any(signal in user_lower for signal in very_long_signals):
        dynamic_max_tokens = 3500
    elif any(signal in user_lower for signal in long_answer_signals):
        dynamic_max_tokens = 1800
    else:
        dynamic_max_tokens = 500

    messages = build_general_chat_messages(user_message, faiss_index, chunks, jd_text, resume_text, fit_analysis, chat_history)

    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=dynamic_max_tokens,
        stream=True,
    )

    was_truncated = False
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
        if chunk.choices[0].finish_reason == "length":
            was_truncated = True

    if was_truncated:
        yield "\n\n---\n⚠️ **This got cut off because it's quite long.** Just say **\"continue\"** and I'll pick up right where I left off."


def chat_response(user_message, faiss_index, chunks, jd_text, resume_text, fit_analysis, chat_history):
    """
    Non-streaming entry point, used for tailoring/cover letter/interview prep
    (which need to return a complete result before being saved to session state
    for later docx export) and as a fallback if streaming isn't used.

    Returns (answer_text, result_type) where result_type is one of:
    "none", "tailoring", "cover_letter", "interview_prep"
    """
    has_jd = jd_text is not None and len(jd_text.strip()) > 0
    has_resume = resume_text is not None and len(resume_text.strip()) > 0
    has_fit = fit_analysis is not None and len(fit_analysis.strip()) > 0

    if not has_jd:
        return ("I don't have a job description yet. Upload one using the JD attach button "
                "below, or paste it directly into the chat and I'll take it from there."), "none"

    intent = detect_intent(user_message, chat_history)

    needs_resume_and_fit_message = (
        "I'd need both your resume and a calculated fit score first. "
        "Attach your resume if you haven't already, and I'll take it from there."
    )

    if intent == "tailoring":
        if has_resume and has_fit:
            return tailor_resume(resume_text, jd_text, fit_analysis), "tailoring"
        return needs_resume_and_fit_message, "none"

    if intent == "cover_letter":
        if has_resume and has_fit:
            return generate_cover_letter(resume_text, jd_text, fit_analysis), "cover_letter"
        return needs_resume_and_fit_message, "none"

    if intent == "interview_prep":
        if has_resume and has_fit:
            return generate_interview_prep(resume_text, jd_text, fit_analysis), "interview_prep"
        return needs_resume_and_fit_message, "none"

    messages = build_general_chat_messages(user_message, faiss_index, chunks, jd_text, resume_text, fit_analysis, chat_history)

    user_lower = user_message.lower()

    very_long_signals = [
        "roadmap", "plan", "comprehensive", "deep dive", "deepen", "everything about",
        "full guide", "complete guide", "all topics", "cover everything", "study plan",
        "preparation plan", "learning path"
    ]
    long_answer_signals = [
        "questions", "list", "examples", "in detail", "explain", "describe",
        "give me", "interview", "prepare", "scenario", "situation", "all the",
        "write", "draft", "summary", "summarize", "topics"
    ]

    if user_lower.strip() in ["continue", "continue please", "keep going", "go on"]:
        dynamic_max_tokens = 3500
    elif any(signal in user_lower for signal in very_long_signals):
        dynamic_max_tokens = 3500
    elif any(signal in user_lower for signal in long_answer_signals):
        dynamic_max_tokens = 1800
    else:
        dynamic_max_tokens = 500

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=dynamic_max_tokens,
    )

    result = response.choices[0].message.content.strip()
    if response.choices[0].finish_reason == "length":
        result += "\n\n---\n⚠️ **This got cut off because it's quite long.** Just say **\"continue\"** and I'll pick up right where I left off."

    return result, "none"
