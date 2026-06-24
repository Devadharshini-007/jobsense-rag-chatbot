"""
streamlit_app.py - JobSense, complete version with:
- Native chat file attach (paperclip icon)
- Smart document type detection (resume vs JD)
- Streaming general chat responses
- Fit score, tailored resume, cover letter, interview prep
- Properly formatted resume docx export (not generic markdown dump)
- Conversation history with working delete
"""

import streamlit as st
from app.embeddings import process_jd_text
from app.rag_pipeline import (
    calculate_fit_score, chat_response, stream_chat_response, detect_intent
)
from app.pdf_utils import extract_text_from_pdf
from app.docx_export import create_docx_from_text
from app.resume_docx import create_resume_docx, extract_candidate_name
from app.conversation_store import (
    init_db, save_conversation, list_conversations,
    load_conversation, delete_conversation, generate_title_from_history
)

init_db()

st.set_page_config(
    page_title="JobSense",
    page_icon="🧠",
    layout="wide"
)

defaults = {
    "faiss_index": None,
    "chunks": [],
    "chat_history": [],
    "jd_text_raw": None,
    "resume_text_raw": None,
    "fit_result": None,
    "tailoring_result": None,
    "cover_letter_result": None,
    "interview_prep_result": None,
    "last_generated_type": None,
    "current_conversation_id": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def wants_docx_download(user_text):
    text_lower = user_text.lower()
    docx_keywords = ["docx", "word doc", "word document", ".doc", "download it", "as a doc", "in doc format"]
    return any(keyword in text_lower for keyword in docx_keywords)


def pick_export_content():
    type_to_content = {
        "tailoring": ("Tailored Resume", st.session_state.tailoring_result),
        "cover_letter": ("Cover Letter", st.session_state.cover_letter_result),
        "interview_prep": ("Interview Preparation", st.session_state.interview_prep_result),
        "fit": ("Job Fit Analysis", st.session_state.fit_result),
    }

    last_type = st.session_state.get("last_generated_type")
    if last_type and last_type in type_to_content:
        title, content = type_to_content[last_type]
        if content:
            return title, content

    for key in ["tailoring", "cover_letter", "interview_prep", "fit"]:
        title, content = type_to_content[key]
        if content:
            return title, content

    return None, None


def build_export_docx(title, content):
    if title == "Tailored Resume" and st.session_state.resume_text_raw:
        candidate_name = extract_candidate_name(st.session_state.resume_text_raw)
        return create_resume_docx(content, candidate_name=candidate_name)
    return create_docx_from_text(title, content)


def render_markdown_lite(text):
    lines = text.split("\n")
    html_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("### ") or stripped.startswith("## ") or stripped.startswith("# "):
            heading_text = stripped.lstrip("#").strip()
            html_lines.append(
                "<div style='font-weight: 700; margin-top: 14px; margin-bottom: 4px;'>" + heading_text + "</div>"
            )
        elif stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if all(c.replace("-", "").strip() == "" for c in cells):
                continue
            row_text = " - ".join(c for c in cells if c)
            html_lines.append("&nbsp;&nbsp;• " + row_text)
        else:
            html_lines.append(line)
    joined = "<br>".join(html_lines)
    parts = joined.split("**")
    rebuilt = ""
    is_bold = False
    for part in parts:
        if is_bold:
            rebuilt += "<strong>" + part + "</strong>"
        else:
            rebuilt += part
        is_bold = not is_bold
    return rebuilt


def add_bot_message(text, download=None):
    message = {"role": "assistant", "content": text}
    if download:
        message["download"] = download
    st.session_state.chat_history.append(message)


def add_user_message(text):
    st.session_state.chat_history.append({"role": "user", "content": text})


def persist_conversation():
    if len(st.session_state.chat_history) == 0:
        return
    title = generate_title_from_history(st.session_state.chat_history)
    history_to_save = []
    for msg in st.session_state.chat_history:
        clean_msg = {"role": msg["role"], "content": msg["content"]}
        history_to_save.append(clean_msg)

    new_id = save_conversation(
        st.session_state.current_conversation_id,
        title,
        history_to_save,
        st.session_state.jd_text_raw,
        st.session_state.resume_text_raw,
        st.session_state.fit_result,
        st.session_state.tailoring_result,
    )
    st.session_state.current_conversation_id = new_id


def classify_document_type(filename, text):
    filename_lower = filename.lower()
    resume_filename_hints = ["resume", "cv", "curriculum"]
    jd_filename_hints = ["jd", "job description", "job_description", "jobdescription", "posting", "vacancy"]

    if any(hint in filename_lower for hint in resume_filename_hints):
        return "resume"
    if any(hint in filename_lower for hint in jd_filename_hints):
        return "jd"

    text_lower = text.lower()[:2000]
    resume_signals = ["education", "experience", "skills", "objective", "career objective",
                       "projects", "certifications", "linkedin.com/in/", "github.com/"]
    jd_signals = ["responsibilities", "requirements", "we are looking for", "job description",
                  "qualifications", "about the role", "what you'll do", "what you will do",
                  "years of experience required", "apply now"]

    resume_score = sum(1 for signal in resume_signals if signal in text_lower)
    jd_score = sum(1 for signal in jd_signals if signal in text_lower)

    if resume_score > jd_score:
        return "resume"
    elif jd_score > resume_score:
        return "jd"
    else:
        return "resume" if st.session_state.jd_text_raw is not None else "jd"


def process_attached_file(uploaded_file, accompanying_text=None):
    extracted_text = extract_text_from_pdf(uploaded_file)

    file_line = "📎 " + uploaded_file.name
    if accompanying_text and accompanying_text.strip():
        combined_message = file_line + "\n" + accompanying_text.strip()
    else:
        combined_message = file_line

    if len(extracted_text.strip()) < 50:
        add_user_message(combined_message)
        add_bot_message("I couldn't read enough text from that PDF. Could you try pasting the content as text instead?")
        return False

    doc_type = classify_document_type(uploaded_file.name, extracted_text)

    if doc_type == "jd" and st.session_state.jd_text_raw is None:
        with st.spinner("Reading job description..."):
            faiss_index, chunks = process_jd_text(extracted_text)
            st.session_state.faiss_index = faiss_index
            st.session_state.chunks = chunks
            st.session_state.jd_text_raw = extracted_text
        add_user_message(combined_message)

        if st.session_state.resume_text_raw is not None:
            with st.spinner("Comparing your resume against the job..."):
                fit_result = calculate_fit_score(st.session_state.resume_text_raw, extracted_text)
                st.session_state.fit_result = fit_result
                st.session_state.last_generated_type = "fit"
            add_bot_message(
                "Got it - I've read through the job description. Here's how you stack up:\n\n" + fit_result +
                "\n\n---\nWant me to tailor your resume, write a cover letter, or help you prep for the interview?"
            )
            return True
        else:
            add_bot_message(
                "Got it - I've read through the job description. Ask me anything about it, "
                "or attach your resume next and I'll check your fit."
            )
            return False

    elif doc_type == "resume" and st.session_state.jd_text_raw is not None and st.session_state.resume_text_raw is None:
        st.session_state.resume_text_raw = extracted_text
        add_user_message(combined_message)
        with st.spinner("Comparing your resume against the job..."):
            fit_result = calculate_fit_score(extracted_text, st.session_state.jd_text_raw)
            st.session_state.fit_result = fit_result
            st.session_state.last_generated_type = "fit"
        add_bot_message(
            "Here's how you stack up against this role:\n\n" + fit_result +
            "\n\n---\nWant me to tailor your resume, write a cover letter, or help you prep for the interview?"
        )
        return True

    elif doc_type == "resume" and st.session_state.jd_text_raw is None:
        st.session_state.resume_text_raw = extracted_text
        add_user_message(combined_message)
        add_bot_message(
            "Got it - I've saved your resume. Now attach the job description and I'll check your fit for it."
        )
        return False

    else:
        st.session_state.resume_text_raw = extracted_text
        add_user_message(combined_message)
        with st.spinner("Re-checking your fit with this updated resume..."):
            fit_result = calculate_fit_score(extracted_text, st.session_state.jd_text_raw)
            st.session_state.fit_result = fit_result
            st.session_state.last_generated_type = "fit"
        add_bot_message("Updated! Here's your new fit analysis:\n\n" + fit_result)
        return True


st.title("🧠 JobSense")
st.caption("Your AI co-pilot for understanding any job and your fit for it.")

with st.sidebar:
    st.markdown("### JobSense")
    if st.button("➕ New Conversation", use_container_width=True):
        for key, value in defaults.items():
            st.session_state[key] = value
        st.rerun()

    st.divider()
    st.markdown("**History**")

    saved_conversations = list_conversations()
    if not saved_conversations:
        st.caption("No saved conversations yet.")
    else:
        for conv in saved_conversations:
            col_load, col_delete = st.columns([5, 1])
            with col_load:
                is_current = conv["id"] == st.session_state.current_conversation_id
                label = ("📍 " if is_current else "") + conv["title"]
                if st.button(label, key="load_" + str(conv["id"]), use_container_width=True):
                    loaded = load_conversation(conv["id"])
                    if loaded:
                        st.session_state.chat_history = loaded["chat_history"]
                        st.session_state.jd_text_raw = loaded["jd_text_raw"]
                        st.session_state.resume_text_raw = loaded["resume_text_raw"]
                        st.session_state.fit_result = loaded["fit_result"]
                        st.session_state.tailoring_result = loaded["tailoring_result"]
                        st.session_state.current_conversation_id = loaded["id"]
                        if loaded["jd_text_raw"]:
                            faiss_index, chunks = process_jd_text(loaded["jd_text_raw"])
                            st.session_state.faiss_index = faiss_index
                            st.session_state.chunks = chunks
                        else:
                            st.session_state.faiss_index = None
                            st.session_state.chunks = []
                    st.rerun()
            with col_delete:
                delete_clicked = st.button("🗑", key="delete_" + str(conv["id"]), help="Delete this conversation")
                if delete_clicked:
                    delete_conversation(conv["id"])
                    if conv["id"] == st.session_state.current_conversation_id:
                        for key, value in defaults.items():
                            st.session_state[key] = value
                    st.toast("Conversation deleted")
                    st.rerun()

    st.divider()
    if st.session_state.jd_text_raw:
        st.caption("✅ Job description loaded")
    if st.session_state.resume_text_raw:
        st.caption("✅ Resume loaded")
    if st.session_state.fit_result:
        st.caption("✅ Fit score calculated")

if len(st.session_state.chat_history) == 0:
    add_bot_message(
        "Hi! Attach a job description using the 📎 icon below (or paste it directly), "
        "and I'll help you understand it, check your fit, and close any skill gaps."
    )

left_space, main_col, right_space = st.columns([1, 4, 1])

with main_col:
    for idx, message in enumerate(st.session_state.chat_history):
        if message["role"] == "assistant":
            st.markdown(
                "<div style='text-align: left; padding: 8px 0 16px 0; font-size: 16px; "
                "color: inherit; max-width: 80%;'>"
                + render_markdown_lite(message["content"]) +
                "</div>",
                unsafe_allow_html=True
            )
            if message.get("download"):
                download_info = message["download"]
                st.download_button(
                    label="⬇ Download " + download_info["title"] + " (.docx)",
                    data=download_info["buffer"],
                    file_name=download_info["filename"],
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                    key="download_" + str(idx),
                )
        else:
            content = message["content"]
            if content.startswith("📎"):
                lines = content.split("\n", 1)
                file_part = "<div style='font-size: 13px; opacity: 0.7; margin-bottom: 4px;'>" + lines[0] + "</div>"
                text_part = render_markdown_lite(lines[1]) if len(lines) > 1 else ""
                bubble_inner = file_part + text_part
            else:
                bubble_inner = render_markdown_lite(content)

            st.markdown(
                "<div style='display: flex; justify-content: flex-end; padding: 8px 0;'>"
                "<div style='background-color: #e8e8e8; color: #1a1a1a; padding: 10px 16px; "
                "border-radius: 18px; max-width: 75%; font-size: 16px; text-align: left;'>"
                + bubble_inner +
                "</div></div>",
                unsafe_allow_html=True
            )

prompt = st.chat_input(
    "Message JobSense...",
    accept_file=True,
    file_type=["pdf"],
)

if prompt:
    has_files = bool(prompt["files"])
    has_text = bool(prompt.text and prompt.text.strip())

    if has_files:
        text_to_attach = prompt.text.strip() if has_text else None
        for i, uploaded_file in enumerate(prompt["files"]):
            text_for_this_file = text_to_attach if i == 0 else None
            process_attached_file(uploaded_file, accompanying_text=text_for_this_file)
        persist_conversation()
        st.rerun()

    elif has_text:
        user_text = prompt.text.strip()
        add_user_message(user_text)

        if st.session_state.jd_text_raw is None and len(user_text) > 300:
            with main_col:
                with st.spinner("Reading job description..."):
                    faiss_index, chunks = process_jd_text(user_text)
                    st.session_state.faiss_index = faiss_index
                    st.session_state.chunks = chunks
                    st.session_state.jd_text_raw = user_text
            add_bot_message(
                "Thanks, I've read through that job description. Ask me anything about it, "
                "or attach/paste your resume next and I'll check your fit."
            )
            persist_conversation()
            st.rerun()

        elif st.session_state.jd_text_raw is not None and st.session_state.resume_text_raw is None and len(user_text) > 300:
            st.session_state.resume_text_raw = user_text
            with main_col:
                with st.spinner("Comparing your resume against the job..."):
                    fit_result = calculate_fit_score(user_text, st.session_state.jd_text_raw)
                    st.session_state.fit_result = fit_result
                    st.session_state.last_generated_type = "fit"
            add_bot_message(
                "Thanks! Here's how you stack up against this role:\n\n" + fit_result +
                "\n\n---\nWant me to tailor your resume, write a cover letter, or help you prep for the interview?"
            )
            persist_conversation()
            st.rerun()

        else:
            if wants_docx_download(user_text):
                title, content = pick_export_content()
                if content:
                    docx_buffer = build_export_docx(title, content)
                    download_data = {
                        "buffer": docx_buffer.getvalue(),
                        "filename": title.replace(" ", "_") + ".docx",
                        "title": title,
                    }
                    add_bot_message(
                        "Here's your " + title.lower() + " as a Word document - download it below.",
                        download=download_data
                    )
                else:
                    add_bot_message(
                        "I don't have anything to export yet. Attach a JD and resume first, "
                        "or ask me something so there's content to download."
                    )
                persist_conversation()
                st.rerun()

            else:
                intent = detect_intent(user_text, st.session_state.chat_history[:-1])

                if intent in ("tailoring", "cover_letter", "interview_prep"):
                    with main_col:
                        with st.spinner("Working on it..."):
                            answer, result_type = chat_response(
                                user_text,
                                st.session_state.faiss_index,
                                st.session_state.chunks,
                                st.session_state.jd_text_raw,
                                st.session_state.resume_text_raw,
                                st.session_state.fit_result,
                                st.session_state.chat_history[:-1]
                            )
                            if result_type == "tailoring":
                                st.session_state.tailoring_result = answer
                                st.session_state.last_generated_type = "tailoring"
                            elif result_type == "cover_letter":
                                st.session_state.cover_letter_result = answer
                                st.session_state.last_generated_type = "cover_letter"
                            elif result_type == "interview_prep":
                                st.session_state.interview_prep_result = answer
                                st.session_state.last_generated_type = "interview_prep"

                    if result_type == "tailoring":
                        add_bot_message(
                            "I've put together your tailored resume - download it below to see the "
                            "full formatted version."
                        )
                        candidate_name = extract_candidate_name(st.session_state.resume_text_raw)
                        docx_buffer = create_resume_docx(answer, candidate_name=candidate_name)
                        st.session_state.chat_history[-1]["download"] = {
                            "buffer": docx_buffer.getvalue(),
                            "filename": "Tailored_Resume.docx",
                            "title": "Tailored Resume",
                        }
                    else:
                        add_bot_message(answer)

                    persist_conversation()
                    st.rerun()

                else:
                    with main_col:
                        placeholder = st.empty()
                        full_response = ""
                        for text_chunk in stream_chat_response(
                            user_text,
                            st.session_state.faiss_index,
                            st.session_state.chunks,
                            st.session_state.jd_text_raw,
                            st.session_state.resume_text_raw,
                            st.session_state.fit_result,
                            st.session_state.chat_history[:-1]
                        ):
                            full_response += text_chunk
                            placeholder.markdown(
                                "<div style='text-align: left; padding: 8px 0 16px 0; font-size: 16px; "
                                "color: inherit; max-width: 80%;'>"
                                + render_markdown_lite(full_response) + " ▌"
                                + "</div>",
                                unsafe_allow_html=True
                            )
                        placeholder.markdown(
                            "<div style='text-align: left; padding: 8px 0 16px 0; font-size: 16px; "
                            "color: inherit; max-width: 80%;'>"
                            + render_markdown_lite(full_response) +
                            "</div>",
                            unsafe_allow_html=True
                        )
                    add_bot_message(full_response)
                    persist_conversation()
                    st.rerun()
