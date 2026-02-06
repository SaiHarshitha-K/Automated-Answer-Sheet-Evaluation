# app.py
import pandas as pd
from main import process_pdf  # use your existing OCR pipeline
import os
import streamlit as st
from datetime import datetime
from utils.prn_utils import normalize_prn
from streamlit_cookies_manager import CookieManager
from course_report import render_course_report
import time

# UI helpers (keeps app.py smaller + prevents rerun-on-keystroke)
from ui_student_editor import render_students_editor
from subject_report import render_subject_report
from ui_students_management import render_students_management
from modules.question_paper_llm import run_question_paper_llm_flow
from dashboard import render_dashboard





# ✅ ADDED: Cookie setup
st.set_page_config(page_title="OMR Admin Portal", layout="wide")
cookies = CookieManager(prefix="omr_admin_")

if not cookies.ready():
    st.stop()

from db_utils import (
    init_db,
    any_admin_exists,
    create_admin,
    validate_admin,
    list_admins,
    add_batch,
    get_batches,
    add_course,
    get_courses,
    add_subject,
    get_subjects,
    save_exam_results,
    load_exam_results,
    update_exam_student_answers,
    update_exam_student_identity,
    get_course_id_for_subject,
    get_course_batch_id,
    find_students_by_last3,
    load_mcq_bank_for_subject,
    import_lab_marks_from_excel,
    get_lab_marks_for_subject,
    upsert_lab_marks,
    get_lab_marks_map,
    publish_latest_exam_for_subject,
    unpublish_subject,
    is_subject_published,

)


# ------------------------------------------------------------------
# Initial setup
# ------------------------------------------------------------------
init_db()

if "admin_id" not in st.session_state:
    st.session_state.admin_id = None
if "username" not in st.session_state:
    st.session_state.username = None
if "is_superadmin" not in st.session_state:
    st.session_state.is_superadmin = False
if "ocr_results" not in st.session_state:
    # {subject_id: {"answer_key": {...}, "students": {...}}}
    st.session_state.ocr_results = {}

# ✅ ADDED: Force logout flag
if "force_logout" not in st.session_state:
    st.session_state.force_logout = False

# ✅ ADDED: Restore login from cookies
if (
    st.session_state.admin_id is None
    and not st.session_state.get("force_logout", False)
):
    try:
        admin_id = cookies.get("admin_id")
        username = cookies.get("username")
        is_super = cookies.get("is_superadmin")

        if admin_id and username:
            st.session_state.admin_id = int(admin_id)
            st.session_state.username = str(username)
            st.session_state.is_superadmin = (str(is_super) == "1")
    except Exception:
        pass


# ------------------------------------------------------------------
# Login & registration views
# ------------------------------------------------------------------
def show_initial_admin_creation():
    st.title("Create First Admin")
    st.info("No admin exists yet. Create the very first super admin.")

    with st.form("first_admin_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        password2 = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Create admin")

    if submitted:
        if not username or not password:
            st.error("Username and password are required.")
        elif password != password2:
            st.error("Passwords do not match.")
        else:
            if create_admin(username, password, superadmin=True):
                st.success("First admin created. Please log in from the main page.")
            else:
                st.error("Could not create admin (probably duplicate username).")


def show_login():
    st.title("Admin Login")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        res = validate_admin(username, password)
        if res is None:
            st.error("Invalid username or password.")
        else:
            admin_id, is_super = res
            st.session_state.admin_id = admin_id
            st.session_state.username = username
            st.session_state.is_superadmin = is_super
            
            # ✅ ADDED: Save to cookies
            cookies["admin_id"] = str(admin_id)
            cookies["username"] = username
            cookies["is_superadmin"] = "1" if is_super else "0"
            cookies.save()
            
            st.rerun()


# ------------------------------------------------------------------
# Logged-in pages
# ------------------------------------------------------------------
def show_admin_dashboard():
    #st.header(f"Welcome, {st.session_state.username}")
    #st.write("Use the menu on the left to manage admins, batches, courses, and subjects.")
    render_dashboard()

def show_admin_management():
    if not st.session_state.is_superadmin:
        st.warning("Only super admins can create new admins.")
        return

    st.header("Admin Management")

    st.subheader("Existing admins")
    admins = list_admins()
    if not admins:
        st.write("No admins found.")
    else:
        for admin_id, username, is_super in admins:
            super_label = " (super)" if is_super else ""
            st.write(f"- {username}{super_label}")

    st.subheader("Create new admin")
    with st.form("new_admin_form"):
        username = st.text_input("New admin username")
        password = st.text_input("Password", type="password")
        password2 = st.text_input("Confirm password", type="password")
        make_super = st.checkbox("Make super admin")
        submitted = st.form_submit_button("Create admin")

    if submitted:
        if not username or not password:
            st.error("Username and password are required.")
        elif password != password2:
            st.error("Passwords do not match.")
        else:
            if create_admin(username, password, superadmin=make_super):
                st.success("Admin created successfully.")
            else:
                st.error("Could not create admin (maybe username already exists).")


def show_students_management():
    st.header("Students Management")

    # 1) Select batch
    batches = get_batches()
    if not batches:
        st.info("No batches found. Please create a batch first.")
        return
    batch_options = {name: bid for bid, name in batches}

    selected_batch_name = st.selectbox(
        "Select Batch",
        ["-- choose --"] + list(batch_options.keys()),
        key="students_batch_select",
    )
    batch_id = batch_options.get(selected_batch_name)

    if not batch_id:
        st.info("Select a batch to manage students.")
        return

    # 2) Select course
    courses = get_courses(batch_id)
    if not courses:
        st.info("No courses found in this batch. Please create a course first.")
        return
    course_options = {name: cid for cid, name in courses}

    selected_course_name = st.selectbox(
        "Select Course",
        ["-- choose --"] + list(course_options.keys()),
        key="students_course_select",
    )
    course_id = course_options.get(selected_course_name)

    if not course_id:
        st.info("Select a course to manage students.")
        return

    # 3) Render management UI
    render_students_management(
        batch_id=batch_id,
        course_id=course_id,
        batch_name=selected_batch_name,
        course_name=selected_course_name,
    )




def run_ocr_for_subject(subject_id: int, pdf_file, key_file):
    """
    Save uploaded files, run OCR on the PDF, compare with key,
    and store results in session_state. One run per subject.
    """
    #base_dir = "data/uploads"
    #os.makedirs(base_dir, exist_ok=True)

    base_dir = os.path.join(
        "data",
        "uploads",
        f"subject_{subject_id}",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )

    os.makedirs(base_dir, exist_ok=True)

    # Save PDF and key to disk
    pdf_path = os.path.join(base_dir, f"subject_{subject_id}_answers.pdf")
    key_path = os.path.join(base_dir, f"subject_{subject_id}_key.xlsx")

    with open(pdf_path, "wb") as f:
        f.write(pdf_file.getbuffer())

    with open(key_path, "wb") as f:
        f.write(key_file.getbuffer())

    # Read key Excel: first row headers, then Qno & option
    df = pd.read_excel(key_path)
    if df.shape[1] < 2:
        st.error("Key Excel must have at least two columns (Qno, Option).")
        return

    q_col = df.columns[0]
    a_col = df.columns[1]

    key_map = {}
    for _, row in df.iterrows():
        q_val = row[q_col]
        opt_val = row[a_col]
        if pd.isna(q_val):
            continue
        try:
            q_no = int(q_val)
        except ValueError:
            continue
        opt_str = "" if pd.isna(opt_val) else str(opt_val).strip().upper()
        key_map[q_no] = opt_str

    # Run your OCR pipeline for THIS pdf only
    pages = process_pdf(pdf_path)

    # Build student-level structures
    students = {}
    total_questions = len(key_map)

    for page in pages:
        name = (page.get("name") or "").strip() or "Unknown"
        #prn = (page.get("prn") or "").strip() or "UNKNOWN"
        raw_prn = (page.get("prn") or "").strip()
        prn = normalize_prn(raw_prn)
        answers = page.get("answers", {})
        img_path = page.get("image_path")

        details = []
        score = 0

        for q_no, correct_opt in key_map.items():
            raw_ans = answers.get(q_no, "")
            ans_norm = "" if raw_ans is None else str(raw_ans).strip().upper()

            is_blank = ans_norm == ""
            is_correct = (not is_blank) and (ans_norm == correct_opt)

            if is_correct:
                score += 1

            details.append(
                {
                    "question": q_no,
                    "student_answer": ans_norm if ans_norm else "(blank)",
                    "key_answer": correct_opt,
                    "is_correct": is_correct,
                    "is_blank": is_blank,
                }
            )

        if prn not in students:
            students[prn] = []

        students[prn].append({
            "name": name,
            "prn": prn,
            "score": score,
            "total": total_questions,
            "details": details,
            "image_path": img_path,
        })


    # persist everything in MySQL
    exam_id = save_exam_results(subject_id, pdf_path, key_path, students)

    # reload from DB (so structure matches what load_exam_results returns)
    result = load_exam_results(subject_id)
    if result is not None:
        st.session_state.ocr_results[subject_id] = result



def show_structure_management():
    st.header("Batches → Courses → Subjects")

    # ===================== BATCHES =====================
    st.subheader("Batches")
    with st.form("batch_form"):
        batch_name = st.text_input("New batch name (e.g., Aug 2025)")
        submitted_batch = st.form_submit_button("Add batch")

    if submitted_batch:
        if not batch_name.strip():
            st.error("Batch name cannot be empty.")
        elif add_batch(batch_name.strip()):
            st.success("Batch added.")
        else:
            st.error("Could not add batch (maybe name already exists).")

    batches = get_batches()
    batch_options = {name: bid for bid, name in batches}
    selected_batch_name = st.selectbox(
        "Select batch", ["-- choose --"] + list(batch_options.keys())
    )
    selected_batch_id = batch_options.get(selected_batch_name)

    # ===================== COURSES =====================
    selected_course_id = None
    if selected_batch_id:
        st.subheader(f"Courses in {selected_batch_name}")
        with st.form("course_form"):
            course_name = st.text_input("New course name (e.g., DBDA, DAC)")
            submitted_course = st.form_submit_button("Add course")

        if submitted_course:
            if not course_name.strip():
                st.error("Course name cannot be empty.")
            elif add_course(selected_batch_id, course_name.strip()):
                st.success("Course added.")
            else:
                st.error("Could not add course.")

        courses = get_courses(selected_batch_id)
        course_options = {name: cid for cid, name in courses}
        selected_course_name = st.selectbox(
            "Select course", ["-- choose --"] + list(course_options.keys())
        )
        selected_course_id = course_options.get(selected_course_name)

        # ===================== COURSE REPORT =====================
    if selected_course_id:
        st.markdown("---")
        st.subheader("📌 Course Report")

        pass_percent = st.number_input(
            "Pass % threshold",
            min_value=0.0,
            max_value=100.0,
            value=35.0,
            step=1.0,
            key=f"course_pass_{selected_course_id}",
        )

        if st.button("📄 Generate Course Report", key=f"gen_course_report_{selected_course_id}"):
            render_course_report(course_id=selected_course_id, pass_percent=pass_percent)


    # ===================== SUBJECTS =====================
    selected_subject_id = None
    if selected_course_id:
        st.subheader(f"Subjects in {selected_course_name}")
        with st.form("subject_form"):
            subject_name = st.text_input("New subject name (e.g., Java, Python)")
            submitted_subject = st.form_submit_button("Add subject")

        if submitted_subject:
            if not subject_name.strip():
                st.error("Subject name cannot be empty.")
            elif add_subject(selected_course_id, subject_name.strip()):
                st.success("Subject added.")
            else:
                st.error("Could not add subject.")

        subjects = get_subjects(selected_course_id)
        subject_options = {name: sid for sid, name in subjects}
        selected_subject_name = st.selectbox(
            "Select subject", ["-- choose --"] + list(subject_options.keys())
        )
        selected_subject_id = subject_options.get(selected_subject_name)

    st.markdown("---")

    # ===================== OCR UPLOAD =====================
    if not selected_subject_id:
        st.info("Select a subject to upload answer PDFs and keys.")
        return
    st.subheader("📄 Question Paper → LLM Extract (Save to DB)")

    qp_pdf = st.file_uploader(
        "Upload Question Paper (PDF)",
        type=["pdf"],
        key=f"qp_pdf_{selected_subject_id}",
    )

    qp_answer_key = st.file_uploader(
        "Upload Answer Key (Excel) (optional)",
        type=["xlsx", "xls"],
        key=f"qp_ak_{selected_subject_id}",
    )

    if st.button("Run LLM & Save Questions", key=f"run_llm_save_{selected_subject_id}"):
        if not qp_pdf:
            st.error("Please upload question paper PDF.")
        else:
            with st.spinner("Running OCR + LLM..."):
                result = run_question_paper_llm_flow(
                    subject_id=selected_subject_id,  # ✅ VERY IMPORTANT: use selected_subject_id
                    qp_pdf_file=qp_pdf,
                    answer_key_file=qp_answer_key,
                    uploads_dir=os.path.join("uploads", "question_papers"),
                )

            st.success(
                f"Saved! Question Paper ID: {result['question_paper_id']} | "
                f"Total Questions saved: {result['inserted_count']}"
            )

            st.rerun()



    # Always show latest saved questions for this subject (if any)
    # ============================================================================
    # IMPROVED DISPLAY CODE FOR app.py
    # Replace the section in show_structure_management() that displays saved questions
    # (Around lines 466-481)
    # ============================================================================

    # Always show latest saved questions for this subject (if any)
    mcq_data_existing = load_mcq_bank_for_subject(selected_subject_id)

    if mcq_data_existing and mcq_data_existing.get("questions"):
        with st.expander("📘 View Saved Questions & Explanations", expanded=False):
            st.info(f"Total Questions: {len(mcq_data_existing['questions'])}")
            
            for q in mcq_data_existing["questions"]:
                # Question header with number
                st.markdown(f"### 📝 Question {q['question_no']}")
                
                # Question text in a highlighted box
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 10px;">
                    <strong>Q:</strong> {q['question_text']}
                </div>
                """, unsafe_allow_html=True)
                
                # Options with color coding
                #correct_opt = q.get('correct_option', '').upper()
                correct_opt = (q.get('correct_option') or '').upper()
                
                # Helper function to style options
                def render_option(letter, text, is_correct, explanation):
                    if is_correct:
                        # Green background for correct answer
                        st.markdown(f"""
                        <div style="background-color: #d4edda; padding: 10px; border-radius: 5px; margin: 5px 0; border-left: 4px solid #28a745;">
                            <strong style="color: #155724;">✓ Option {letter}) {text}</strong><br>
                            <span style="color: #155724; font-size: 0.9em;">
                                <strong>Explanation:</strong> {explanation}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Light red background for incorrect answer
                        st.markdown(f"""
                        <div style="background-color: #f8d7da; padding: 10px; border-radius: 5px; margin: 5px 0; border-left: 4px solid #dc3545;">
                            <strong style="color: #721c24;">✗ Option {letter}) {text}</strong><br>
                            <span style="color: #721c24; font-size: 0.9em;">
                                <strong>Why incorrect:</strong> {explanation}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Render all options
                render_option(
                    "A", 
                    q.get('option_a', ''), 
                    correct_opt == 'A',
                    q.get('why_correct', '') if correct_opt == 'A' else q.get('why_a_wrong', '')
                )
                
                render_option(
                    "B", 
                    q.get('option_b', ''), 
                    correct_opt == 'B',
                    q.get('why_correct', '') if correct_opt == 'B' else q.get('why_b_wrong', '')
                )
                
                render_option(
                    "C", 
                    q.get('option_c', ''), 
                    correct_opt == 'C',
                    q.get('why_correct', '') if correct_opt == 'C' else q.get('why_c_wrong', '')
                )
                
                render_option(
                    "D", 
                    q.get('option_d', ''), 
                    correct_opt == 'D',
                    q.get('why_correct', '') if correct_opt == 'D' else q.get('why_d_wrong', '')
                )
                
                # Answer summary
                if correct_opt:
                    st.success(f"✅ **Correct Answer: Option {correct_opt}**")
                    st.info(f"**Why this is correct:** {q.get('why_correct', 'No explanation available')}")
                
                st.markdown("---")


    st.markdown("---")
    st.subheader(f"OCR & Evaluation — {selected_subject_name}")

    with st.form("upload_form"):
        pdf_file = st.file_uploader("Upload answer-sheet PDF", type=["pdf"])
        key_file = st.file_uploader("Upload answer key (Excel)", type=["xlsx", "xls"])
        submitted_upload = st.form_submit_button("Run OCR and Evaluate")



    if submitted_upload:
        if not pdf_file or not key_file:
            st.error("Upload both PDF and Excel key.")
        else:
            run_ocr_for_subject(selected_subject_id, pdf_file, key_file)
            st.success("OCR completed.")

    # ✅ Lab marks upload (independent of OCR, just used for report)
    st.markdown("---")
    st.subheader("🧪 Lab Marks (DB)")

    lab_excel = st.file_uploader(
        "Upload Lab Marks Excel (PRN | LAB_MARKS)",
        type=["xlsx", "xls"],
        key=f"lab_{selected_subject_id}",
    )

    colA, colB = st.columns(2)

    with colA:
        if st.button("⬆️ Import / Update Lab Marks to DB", key=f"import_lab_{selected_subject_id}"):
            if not lab_excel:
                st.error("Please upload Lab Excel first.")
            else:
                res = import_lab_marks_from_excel(
                    subject_id=selected_subject_id,
                    excel_file=lab_excel,
                    updated_by=st.session_state.admin_id,
                )
                st.success(f"Imported/Updated: {res['inserted_or_updated']} | Skipped: {res['skipped']}")
                st.rerun()

    # Show editable table (dropdown-ish + editable)
    # we need course_id to show all course students
    course_id_for_this_subject = get_course_id_for_subject(selected_subject_id)

    rows = get_lab_marks_for_subject(selected_subject_id, course_id=course_id_for_this_subject)

    if not rows:
        st.info("No students / lab rows found yet.")
    else:
        import pandas as pd

        df_lab = pd.DataFrame(rows)

        # Ensure columns exist
        if "marks" not in df_lab.columns:
            df_lab["marks"] = None
        if "name" not in df_lab.columns:
            df_lab["name"] = ""

        st.caption("Edit marks and click Save. Empty marks = NULL.")
        edited = st.data_editor(
            df_lab[["prn", "name", "marks"]],
            disabled=["prn", "name"],
            use_container_width=True,
            key=f"lab_editor_{selected_subject_id}",
        )

        with colB:
            if st.button("💾 Save Edited Lab Marks", key=f"save_lab_{selected_subject_id}"):
                # Update DB for each row
                saved = 0
                for _, r in edited.iterrows():
                    prn = "".join(ch for ch in str(r["prn"]) if ch.isdigit()).strip()

                    marks = r["marks"]
                    ok = upsert_lab_marks(
                        subject_id=selected_subject_id,
                        prn=prn,
                        marks=marks,
                        updated_by=st.session_state.admin_id,
                    )
                    if ok:
                        saved += 1

                st.success(f"Saved {saved} lab marks.")
                st.rerun()



    # ===================== LOAD RESULTS =====================
    subject_result = st.session_state.ocr_results.get(selected_subject_id)
    if subject_result is None:
        subject_result = load_exam_results(selected_subject_id)
        if subject_result:
            st.session_state.ocr_results[selected_subject_id] = subject_result

    if not subject_result:
        st.info("No OCR results yet.")
        return
    st.markdown("---")
    # st.subheader("📢 Post to Students (Publish)")

    # published = is_subject_published(selected_subject_id)

    # colP1, colP2, colP3 = st.columns([1, 1, 2])

    # with colP1:
    #     if st.button("✅ Post / Publish to Students", key=f"publish_{selected_subject_id}"):
    #         res = publish_latest_exam_for_subject(
    #             subject_id=selected_subject_id,
    #             admin_id=st.session_state.admin_id,
    #         )
    #         if res.get("ok"):
    #             st.success(f"Published! (Exam ID: {res['exam_id']})")
    #         else:
    #             st.error(res.get("error", "Publish failed"))
    #         st.rerun()

    # with colP2:
    #     if st.button("❌ Unpublish", key=f"unpublish_{selected_subject_id}"):
    #         ok = unpublish_subject(selected_subject_id)
    #         if ok:
    #             st.warning("Unpublished. Students will not see it now.")
    #         else:
    #             st.info("Already unpublished / not posted yet.")
    #         st.rerun()

    # with colP3:
    #     st.write("Status:", "✅ Published" if published else "⏳ Not posted yet")
    st.subheader("📢 Post to Students (Publish)")

    published = is_subject_published(selected_subject_id)
    colP1, colP2, colP3 = st.columns([1, 1, 2])

    with colP1:
        send_email = st.checkbox(
            "📧 Send email to students",
            value=True,
            key=f"send_email_toggle_{selected_subject_id}",
        )

    with colP2:
        colA, colB = st.columns(2)

        with colA:
            if st.button("✅ Publish", key=f"publish_btn_{selected_subject_id}"):
                res = publish_latest_exam_for_subject(
                    subject_id=selected_subject_id,
                    admin_id=st.session_state.admin_id,
                )

                if res.get("ok"):
                    st.success(f"Published! (Exam ID: {res['exam_id']})")

                    if send_email:
                        from email_service import send_publish_emails_to_students

                        with st.spinner("Sending student emails..."):
                            mail_res = send_publish_emails_to_students(
                                exam_id=int(res["exam_id"]),
                                subject_id=int(selected_subject_id),
                                theory_out_of=40.0,
                                lab_out_of=40.0,
                                attach_txt=True,   # optional attachment
                            )

                        if mail_res.get("ok"):
                            st.success(
                                f"Emails sent: {mail_res['sent']} | "
                                f"Skipped(no email): {mail_res['skipped_no_email']} | "
                                f"Failed: {mail_res['failed']}"
                            )
                            if mail_res.get("errors"):
                                st.warning("Some failures:")
                                for e in mail_res["errors"]:
                                    st.write("-", e)
                        else:
                            st.error("Published but email sending failed.")
                            st.error(mail_res.get("error"))
                else:
                    st.error(res.get("error", "Publish failed"))

                st.rerun()

        with colB:
            if st.button("❌ Unpublish", key=f"unpublish_btn_{selected_subject_id}"):
                ok = unpublish_subject(selected_subject_id)
                if ok:
                    st.warning("Unpublished. Students will not see it now.")
                else:
                    st.info("Already unpublished / not posted yet.")
                st.rerun()

    with colP3:
        st.write("Status:", "✅ Published" if published else "⏳ Not posted yet")


    key_map = subject_result["answer_key"]
    students = subject_result["students"]

    st.subheader("Detected students (including duplicates)")

    # ===================== STUDENTS DISPLAY (FAST, FORM-BASED) =====================
    # - Renders 5 answers per row
    # - Avoids rerun on every keypress by using st.form
    # - Updates ONLY the clicked student attempt and patches session_state in-place
    render_students_editor(
        subject_id=selected_subject_id,
        subject_result=subject_result,
        key_map=key_map,
        update_exam_student_identity=update_exam_student_identity,
        update_exam_student_answers=update_exam_student_answers,
        get_course_id_for_subject=get_course_id_for_subject,
        get_course_batch_id=get_course_batch_id,
        find_students_by_last3=find_students_by_last3,
    )

    st.markdown("---")

    if st.button("📄 Generate Subject Report", key=f"report_{selected_subject_id}"):
        lab_map = get_lab_marks_map(selected_subject_id)

        render_subject_report(
            subject_name=selected_subject_name,
            subject_result=subject_result,
            key_map=key_map,
            lab_marks_map=lab_map,
        )



def show_duplicates():
    st.header("Duplicate PRN Conflicts")

    duplicates = st.session_state.get("duplicates", {})

    if not duplicates:
        st.info("No duplicate PRNs detected so far.")
        return

    for subject_id, prn_map in duplicates.items():
        st.subheader(f"Subject ID: {subject_id}")

        for prn, entries in prn_map.items():
            st.warning(f"Duplicate PRN: {prn}")

            for idx, entry in enumerate(entries, start=1):
                with st.expander(f"Entry {idx} for PRN {prn}"):

                    if entry.get("image_path") and os.path.exists(entry["image_path"]):
                        st.image(entry["image_path"], width=300)

                    new_name = st.text_input(
                        "Correct Name",
                        value=entry.get("name", ""),
                        key=f"dup_name_{subject_id}_{prn}_{idx}",
                    )

                    new_prn = st.text_input(
                        "Correct PRN",
                        value=prn,
                        key=f"dup_prn_{subject_id}_{prn}_{idx}",
                    )

                    if st.button(
                        "Resolve Duplicate",
                        key=f"resolve_{subject_id}_{prn}_{idx}",
                    ):
                        st.success(
                            "Resolved. Go back to the subject page to see updates."
                        )


# ✅ ADDED: Logout function
def do_logout():
    """Handle logout properly"""
    st.session_state.force_logout = True
    
    # Clear cookies
    try:
        cookies["admin_id"] = ""
        cookies["username"] = ""
        cookies["is_superadmin"] = ""
        cookies.save()
    except Exception:
        pass
    
    # Clear session
    st.session_state.admin_id = None
    st.session_state.username = None
    st.session_state.is_superadmin = False
    
    st.rerun()


# ------------------------------------------------------------------
# Layout / Routing
# ------------------------------------------------------------------
def main():
    # ✅ MODIFIED: Moved st.set_page_config to top (already done above)
    
    # ✅ ADDED: Reset force_logout flag when logged out
    if st.session_state.admin_id is None:
        st.session_state.force_logout = False
        
        # no admin logged in
        if not any_admin_exists():
            show_initial_admin_creation()
        else:
            show_login()
        return

    # logged in
    st.sidebar.write(f"Logged in as: **{st.session_state.username}**")
    menu_choice = st.sidebar.radio(
        "Menu",
        ("Dashboard", "Admins","Students", "Batches / Courses / Subjects","Logout"),
    )

    if menu_choice == "Dashboard":
        show_admin_dashboard()
    elif menu_choice == "Admins":
        show_admin_management()
    elif menu_choice == "Students":
        show_students_management()
    elif menu_choice == "Batches / Courses / Subjects":
        show_structure_management()
    #elif menu_choice == "Duplicates / Conflicts":
    #    show_duplicates()
    elif menu_choice == "Logout":
        do_logout()  # ✅ CHANGED: Use proper logout function


if __name__ == "__main__":
    main()