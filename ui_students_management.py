import pandas as pd
import streamlit as st
from student_wise_report import render_student_wise_report

from db_utils import (
    upsert_student,
    bulk_upsert_students_from_df,
    list_students_for_course,
    admin_reset_student_password_to_prn,
    auto_link_exam_students_by_last3,
    get_latest_exam_id_for_course,
)


def render_students_management(
    *,
    batch_id: int,
    course_id: int,
    batch_name: str,
    course_name: str,
):
    """
    Admin UI:
    - Add single student
    - Upload Excel to add students
    - View students list
    - Reset password to PRN
    """

    st.subheader(f"👩‍🎓 Students — {course_name} ({batch_name})")

    # ------------------------------------------------------------------
    # 1️⃣ ADD SINGLE STUDENT
    # ------------------------------------------------------------------
    with st.expander("➕ Add Single Student", expanded=False):
        with st.form("add_single_student_{course_id}"):
            prn = st.text_input("PRN *")
            name = st.text_input("Student Name *")
            phone = st.text_input("Phone (optional)")
            email = st.text_input("Email (optional)")

            submitted = st.form_submit_button("Add / Update Student")

        if submitted:
            ok = upsert_student(
                prn=prn,
                name=name,
                phone=phone or None,
                email=email or None,
                batch_id=batch_id,
                course_id=course_id,
            )

            if ok:
                st.success("Student added / updated successfully.")
                st.info("Login created → **User ID = PRN, Password = PRN**")
                st.rerun()
            else:
                st.error("Failed. PRN and Name are mandatory.")

    # ------------------------------------------------------------------
    # 2️⃣ BULK UPLOAD FROM EXCEL
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 2️⃣ BULK UPLOAD FROM EXCEL
    # ------------------------------------------------------------------
    with st.expander("📥 Upload Students Excel", expanded=False):

        # 1) dynamic uploader key (changes after successful import)
        base_key = f"students_excel_{course_id}"
        key_version = st.session_state.get(f"{base_key}_v", 0)
        uploader_key = f"{base_key}_{key_version}"

        excel_file = st.file_uploader(
            "Upload Excel",
            type=["xlsx", "xls"],
            key=uploader_key,
        )

        processed_key = f"{base_key}_processed_sig"

        if excel_file is not None:
            file_sig = f"{excel_file.name}_{excel_file.size}"

            # prevent duplicate processing on reruns
            if st.session_state.get(processed_key) == file_sig:
                st.info("✅ File already processed. Upload a new file to import again.")
            else:
                try:
                    df = pd.read_excel(excel_file, dtype=str)  # ✅ keeps 001 as "001"
                    df = df.fillna("")                         # optional safety


                    result = bulk_upsert_students_from_df(
                        df,
                        batch_id=batch_id,
                        course_id=course_id,
                    )

                    st.success(
                        f"Inserted / Updated: {result['inserted_or_updated']} | "
                        f"Skipped: {result['skipped']}"
                    )

                    if result["errors"]:
                        st.warning("Some rows had issues:")
                        for e in result["errors"]:
                            st.write(e)

                    # mark processed
                    st.session_state[processed_key] = file_sig

                    # ✅ bump uploader key version -> uploader resets next rerun (no exception)
                    st.session_state[processed_key] = file_sig
                    st.session_state[f"{base_key}_v"] = key_version + 1

                    # ✅ auto-link OCR results for this course (latest exam)
                    latest_exam_id = get_latest_exam_id_for_course(course_id)

                    if latest_exam_id:
                        link_res = auto_link_exam_students_by_last3(latest_exam_id)
                        st.info(
                            f"🔗 OCR Auto-link (latest exam): "
                            f"Updated={link_res['updated']} | Conflicts={link_res['conflicts']} | NoMatch={link_res['no_match']}"
                        )
                    else:
                        st.warning("No exam found for this course yet, so OCR auto-link skipped.")

                    st.rerun()


                except Exception as e:
                    st.error(f"Failed to process Excel: {e}")


    # ------------------------------------------------------------------
    # 3️⃣ LIST STUDENTS
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📋 Students List")

    students = list_students_for_course(course_id)

    if not students:
        st.info("No students added yet.")
        return

    df_view = pd.DataFrame(students)[
        ["prn", "name", "phone", "email", "created_at"]
    ]

    st.dataframe(df_view, use_container_width=True)

    # ------------------------------------------------------------------
    # 5️⃣ STUDENT-WISE REPORT (NEW)
    # ------------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📄 Student-wise Report")

    # build options from already loaded students
    prn_to_name = {str(x["prn"]): str(x.get("name", "")) for x in students}

    student_labels = [f"{p} — {prn_to_name[p]}" for p in prn_to_name.keys()]
    label_to_prn = {lab: lab.split(" — ")[0].strip() for lab in student_labels}

    selected_label = st.selectbox(
        "Select Student (PRN)",
        ["-- choose --"] + student_labels,
        key=f"student_report_select_{course_id}",
    )

    if selected_label != "-- choose --":
        selected_prn = label_to_prn[selected_label]

        if st.button("Generate Student Report", key=f"btn_student_report_{course_id}_{selected_prn}"):
            render_student_wise_report(
                prn=selected_prn,
                course_id=course_id,
                theory_max=40,
                lab_max=40,
                pass_mark=16,
            )


    # ------------------------------------------------------------------
    # 4️⃣ PASSWORD RESET
    # ------------------------------------------------------------------
    with st.expander("🔐 Reset Student Password (Admin)", expanded=False):
        prn_reset = st.text_input("PRN to reset password", key=f"reset_prn_{course_id}")

        if st.button("Reset Password to PRN", key=f"reset_btn_{course_id}"):

            if not prn_reset.strip():
                st.error("PRN required.")
            else:
                ok = admin_reset_student_password_to_prn(prn_reset.strip())
                if ok:
                    st.success("Password reset successfully (Password = PRN).")
                else:
                    st.error("PRN not found.")
