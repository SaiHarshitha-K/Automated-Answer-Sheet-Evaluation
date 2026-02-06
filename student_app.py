import os
import pandas as pd
import streamlit as st
from streamlit_cookies_manager import CookieManager

from db_utils import (
    init_db,
    validate_student,
    get_student_profile,
    get_subjects,
    is_subject_published,
    get_published_exam_id,
    get_student_exam_sheet_and_key,   # image + answers table
    get_student_question_review,      # mcq_bank + exam_answers join
    get_student_report_published_only # ✅ report (published only)
)

# ------------------------------------------------------------
# Page config + Student cookies
# ------------------------------------------------------------
st.set_page_config(page_title="OMR Student Portal", layout="wide")
cookies = CookieManager(prefix="omr_student_")

if not cookies.ready():
    st.stop()

# ------------------------------------------------------------
# Init DB
# ------------------------------------------------------------
init_db()

# ------------------------------------------------------------
# Session defaults
# ------------------------------------------------------------
if "student_prn" not in st.session_state:
    st.session_state.student_prn = None
if "student_name" not in st.session_state:
    st.session_state.student_name = None
if "student_course_id" not in st.session_state:
    st.session_state.student_course_id = None
if "student_batch_id" not in st.session_state:
    st.session_state.student_batch_id = None
if "force_logout_student" not in st.session_state:
    st.session_state.force_logout_student = False

# ------------------------------------------------------------
# Restore login from cookies
# ------------------------------------------------------------
if st.session_state.student_prn is None and not st.session_state.force_logout_student:
    try:
        prn = cookies.get("student_prn")
        name = cookies.get("student_name")
        course_id = cookies.get("student_course_id")
        batch_id = cookies.get("student_batch_id")

        if prn and name:
            st.session_state.student_prn = str(prn)
            st.session_state.student_name = str(name)
            st.session_state.student_course_id = int(course_id) if course_id else None
            st.session_state.student_batch_id = int(batch_id) if batch_id else None
    except Exception:
        pass


# ------------------------------------------------------------
# UI: Login
# ------------------------------------------------------------
def show_student_login():
    st.title("Student Login")

    with st.form("student_login_form"):
        prn = st.text_input("PRN")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        row = validate_student(prn, password)
        if not row:
            st.error("Invalid PRN or password.")
            return

        profile = get_student_profile(prn)
        if not profile:
            st.error("Student exists but profile not found.")
            return

        st.session_state.student_prn = profile["prn"]
        st.session_state.student_name = profile["name"]
        st.session_state.student_course_id = profile.get("course_id")
        st.session_state.student_batch_id = profile.get("batch_id")

        cookies["student_prn"] = str(profile["prn"])
        cookies["student_name"] = str(profile["name"])
        cookies["student_course_id"] = str(profile.get("course_id") or "")
        cookies["student_batch_id"] = str(profile.get("batch_id") or "")
        cookies.save()

        st.rerun()


# ------------------------------------------------------------
# UI: Logout
# ------------------------------------------------------------
def do_student_logout():
    st.session_state.force_logout_student = True

    try:
        cookies["student_prn"] = ""
        cookies["student_name"] = ""
        cookies["student_course_id"] = ""
        cookies["student_batch_id"] = ""
        cookies.save()
    except Exception:
        pass

    st.session_state.student_prn = None
    st.session_state.student_name = None
    st.session_state.student_course_id = None
    st.session_state.student_batch_id = None

    st.rerun()


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _resolve_image_path(p: str | None) -> str | None:
    if not p:
        return None
    p = str(p).strip()
    if not p:
        return None
    return os.path.abspath(p) if not os.path.isabs(p) else p


def _render_question_block(row: dict):
    """
    Rules:
    ✅ If student == key:
        - show all options normal
        - ONLY correct option GREEN with why_correct
    ✅ If student != key:
        - student's chosen option RED with why_<chosen>_wrong
        - correct option GREEN with why_correct
        - other options normal
    ✅ If blank:
        - correct option GREEN with why_correct
        - no red option
    """
    qno = row.get("question_no")
    qtext = row.get("question_text") or ""

    options = {
        "A": row.get("option_a") or "",
        "B": row.get("option_b") or "",
        "C": row.get("option_c") or "",
        "D": row.get("option_d") or "",
    }

    student_ans = (row.get("student_answer") or "(blank)").strip().upper()
    key_ans = (row.get("key_answer") or "").strip().upper()

    if student_ans in ["", "NONE", "(BLANK)", "(blank)"]:
        student_ans = "(blank)"

    why_correct = row.get("why_correct") or ""
    why_wrong_map = {
        "A": row.get("why_a_wrong") or "",
        "B": row.get("why_b_wrong") or "",
        "C": row.get("why_c_wrong") or "",
        "D": row.get("why_d_wrong") or "",
    }

    is_blank = (student_ans == "(blank)")
    is_correct = (not is_blank) and (student_ans == key_ans)

    def card(label, body, bg, border):
        return f"""
        <div style="
            background:{bg};
            border-left:6px solid {border};
            padding:14px 14px;
            border-radius:10px;
            margin:10px 0;
        ">
            <div style="font-size:17px; font-weight:700; margin-bottom:8px;">
                {label}
            </div>
            <div style="font-size:16px; line-height:1.5;">
                {body}
            </div>
        </div>
        """

    st.markdown(f"## ✏️ Question {qno}")
    st.markdown(
        f"""
        <div style="background:#f6f7f9; padding:14px; border-radius:10px; font-size:17px;">
            <b>Q:</b> {qtext}
        </div>
        """,
        unsafe_allow_html=True,
    )

    for opt in ["A", "B", "C", "D"]:
        text = options.get(opt, "")
        if not text:
            continue

        # GREEN (correct key option)
        if opt == key_ans:
            st.markdown(
                card(
                    f"✓ Option {opt}) {text}",
                    f"<b>Explanation:</b> {why_correct or '—'}",
                    bg="#d4edda",
                    border="#198754",
                ),
                unsafe_allow_html=True,
            )

        # RED (wrong chosen option)
        elif (not is_blank) and (not is_correct) and (opt == student_ans):
            st.markdown(
                card(
                    f"✗ Option {opt}) {text}",
                    f"<b>Why incorrect:</b> {why_wrong_map.get(opt, '') or '—'}",
                    bg="#f8d7da",
                    border="#dc3545",
                ),
                unsafe_allow_html=True,
            )

        # Normal
        else:
            st.markdown(
                f"""
                <div style="
                    background:#ffffff;
                    border:1px solid #e6e6e6;
                    padding:12px 14px;
                    border-radius:10px;
                    margin:10px 0;
                    font-size:16px;
                ">
                    <b>Option {opt})</b> {text}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Summary line
    if is_correct:
        st.success(f"✅ Correct Answer: Option {key_ans}")
    elif is_blank:
        st.warning("⚪ You left this question blank.")
        st.success(f"✅ Correct Answer: Option {key_ans}")
    else:
        st.error(f"❌ Your Answer: Option {student_ans}")
        st.success(f"✅ Correct Answer: Option {key_ans}")

    st.markdown("---")


# ------------------------------------------------------------
# UI: Student Dashboard
# ------------------------------------------------------------
def show_student_dashboard():
    profile = get_student_profile(st.session_state.student_prn)

    st.header(f"Hi, {profile['name']}")
    st.write(f"**PRN:** {profile['prn']}")
    st.write(f"**Course:** {profile.get('course_name') or '-'}")
    st.write(f"**Batch:** {profile.get('batch_name') or '-'}")

    # ✅ MUST define course_id BEFORE report call
    course_id = profile.get("course_id")
    if not course_id:
        st.info("Course is not linked yet. Ask admin to assign course.")
        return

    st.markdown("---")

    # --------------------------------------------------------
    # ✅ MY REPORT (Published subjects only) — ABOVE subject select
    # --------------------------------------------------------
    st.subheader("📄 My Report (Published Subjects Only)")

    report = get_student_report_published_only(
        prn=profile["prn"],
        course_id=course_id,
        theory_max=40.0,
        lab_max=40.0,
        pass_total_min=16.0,
    )

    rep_rows = report.get("rows") or []
    summ = report.get("summary") or {}

    if not rep_rows:
        st.info(summ.get("message") or "No published subjects report available yet.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Overall %", f"{summ.get('overall_percent', 0)}%")
        c2.metric("Total", f"{summ.get('overall_total', 0)} / {summ.get('overall_max', 0)}")
        c3.metric("Published Subjects", f"{summ.get('subjects_count', 0)}")

        if summ.get("failed_subjects"):
            st.error("❌ Failed subjects: " + ", ".join(summ["failed_subjects"]))
        if summ.get("below_20_subjects"):
            st.warning("⚠️ Study hard (<20): " + ", ".join(summ["below_20_subjects"]))

        df_rep = pd.DataFrame(rep_rows)

        def fmt_absent(v, label):
            return label if v is None else v

        df_rep["Theory (40)"] = df_rep["theory_marks"].apply(lambda x: fmt_absent(x, "Absent"))
        df_rep["Lab (40)"] = df_rep["lab_marks"].apply(lambda x: fmt_absent(x, "Not uploaded"))
        df_rep["Total (80)"] = df_rep["total"].apply(lambda x: fmt_absent(x, "-"))
        df_rep["Class Avg (Total)"] = df_rep["class_avg_total"].apply(lambda x: fmt_absent(x, "-"))
        df_rep["Rank"] = df_rep.apply(
            lambda r: "-" if r.get("rank_total") is None or r.get("class_size") is None else f"{r['rank_total']} / {r['class_size']}",
            axis=1,
        )

        df_show = df_rep[["subject_name", "Theory (40)", "Lab (40)", "Total (80)", "status", "Class Avg (Total)", "Rank"]]
        df_show = df_show.rename(columns={"subject_name": "Subject", "status": "Status"})
        st.dataframe(df_show, use_container_width=True)

    st.markdown("---")

    # --------------------------------------------------------
    # SUBJECT SELECT (published only display continues)
    # --------------------------------------------------------
    st.subheader("Subjects")

    subjects = get_subjects(course_id)
    if not subjects:
        st.info("No subjects added yet for your course.")
        return

    subject_options = {name: sid for sid, name in subjects}
    selected_subject_name = st.selectbox(
        "Select Subject",
        ["-- choose --"] + list(subject_options.keys()),
        key="student_subject_select",
    )

    if selected_subject_name == "-- choose --":
        return

    subject_id = int(subject_options[selected_subject_name])

    if not is_subject_published(subject_id):
        st.warning("This subject is not published by admin yet.")
        return

    exam_id = get_published_exam_id(subject_id)
    if not exam_id:
        st.warning("Published exam not found. Ask admin to publish again.")
        return

    st.success(f"Published ✅ — {selected_subject_name}")

    # --------------------------------------------------------
    # TOP: Image + Answer Key table
    # --------------------------------------------------------
    data = get_student_exam_sheet_and_key(prn=profile["prn"], exam_id=exam_id)
    if not data:
        st.info("No exam record found for your PRN in the published exam.")
        return

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("🖼️ Your Answer Sheet")
        img_path = _resolve_image_path(data.get("image_path"))
        if img_path and os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.info("Answer sheet image not available (image_path missing or file not found).")

        score = data.get("score", 0)
        total_q = data.get("total_questions", 0)
        percent = round((score / total_q * 100.0), 2) if total_q else 0.0
        st.markdown(f"**Score:** {score} / {total_q}  |  **Percent:** {percent}%")

    with col_right:
        st.subheader("🧾 Answer Key (Question-wise)")

        rows = data.get("answers") or []
        if not rows:
            st.info("No answers saved for this exam.")
            return

        df = pd.DataFrame(rows).rename(
            columns={
                "question_no": "Q.No",
                "student_answer": "Your Answer",
                "key_answer": "Key",
                "is_correct": "Correct?",
                "is_blank": "Blank?",
            }
        )

        show_cols = ["Q.No", "Your Answer", "Key", "Correct?", "Blank?"]
        df = df[show_cols]
        st.dataframe(df, use_container_width=True, height=650)

    # --------------------------------------------------------
    # BELOW: Question review (full width)
    # --------------------------------------------------------
    st.markdown("---")
    st.subheader("📘 Question-wise Review (Options + Explanations)")

    try:
        review_rows = get_student_question_review(profile["prn"], subject_id, exam_id=exam_id)
    except TypeError:
        review_rows = get_student_question_review(profile["prn"], subject_id)

    if not review_rows:
        st.info("Question review not available (MCQ bank not generated or answers missing).")
        return

    filter_mode = st.selectbox("Filter", ["All", "Only Wrong", "Only Correct", "Only Blank"])

    for r in review_rows:
        s_ans = (r.get("student_answer") or "(blank)").strip().upper()
        k_ans = (r.get("key_answer") or "").strip().upper()
        blank = s_ans in ["", "NONE", "(BLANK)", "(blank)"]
        correct = (not blank) and (s_ans == k_ans)

        if filter_mode == "Only Wrong" and (correct or blank):
            continue
        if filter_mode == "Only Correct" and not correct:
            continue
        if filter_mode == "Only Blank" and not blank:
            continue

        title = f"Q{r.get('question_no')}  •  {'✅ Correct' if correct else ('⚪ Blank' if blank else '❌ Wrong')}"
        with st.expander(title, expanded=False):
            _render_question_block(r)


# ------------------------------------------------------------
# Main routing
# ------------------------------------------------------------
def main():
    if st.session_state.student_prn is None:
        st.session_state.force_logout_student = False
        show_student_login()
        return

    st.sidebar.write(f"Logged in as: **{st.session_state.student_name}**")
    menu = st.sidebar.radio("Menu", ("Dashboard", "Logout"))

    if menu == "Dashboard":
        show_student_dashboard()
    elif menu == "Logout":
        do_student_logout()


if __name__ == "__main__":
    main()
