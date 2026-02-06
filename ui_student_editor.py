from __future__ import annotations

import os
from typing import Dict, Any, Callable, List
import streamlit as st


def inject_compact_css() -> None:
    st.markdown(
        """
        <style>
          div.block-container { padding-top: 1rem; }
          div[data-testid="stVerticalBlock"] { gap: 0.35rem; }

          div[data-baseweb="input"] input {
            padding: 0.25rem 0.45rem !important;
            font-size: 18px !important;
          }

          .stCaption { margin-bottom: 0.15rem; }
          section[data-testid="stExpander"] details summary { font-size: 1.02rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _digits_only(s: str) -> str:
    return "".join(ch for ch in (s or "") if ch.isdigit())


def _extract_last3_digits(prn: str) -> str | None:
    d = _digits_only(prn)
    if not d:
        return None
    return d[-3:] if len(d) >= 3 else d.zfill(3)



def _update_session_identity(
    *,
    subject_result: Dict[str, Any],
    old_prn: str,
    entry_index: int,
    new_prn: str,
    new_name: str,
) -> None:
    students = subject_result.get("students", {})

    if old_prn not in students:
        return
    if entry_index < 0 or entry_index >= len(students[old_prn]):
        return

    entry = students[old_prn][entry_index]
    entry["name"] = new_name
    entry["prn"] = new_prn

    if new_prn != old_prn:
        moved = students[old_prn].pop(entry_index)
        if not students[old_prn]:
            students.pop(old_prn, None)
        students.setdefault(new_prn, []).append(moved)


def _update_session_answers(
    *,
    subject_result: Dict[str, Any],
    prn: str,
    entry_index: int,
    new_score: int,
    new_details: list,
) -> None:
    students = subject_result.get("students", {})
    if prn not in students:
        return
    if entry_index < 0 or entry_index >= len(students[prn]):
        return

    entry = students[prn][entry_index]
    entry["score"] = int(new_score)
    entry["details"] = new_details


def render_students_editor(
    *,
    subject_id: int,
    subject_result: Dict[str, Any],
    key_map: Dict[int, str],
    update_exam_student_identity: Callable[[int, str, str], None],
    update_exam_student_answers: Callable[[int, int, list], None],

    # DB helpers passed from app.py
    get_course_id_for_subject: Callable[[int], int | None],
    get_course_batch_id: Callable[[int], int | None],
    find_students_by_last3: Callable[..., List[dict]],
) -> None:
    inject_compact_css()

    students = subject_result.get("students", {})
    q_numbers = sorted(key_map.keys())

    # 1) Flatten attempts
    attempts = []
    for prn, entries in students.items():
        for attempt_idx, info in enumerate(entries):
            attempts.append(
                {
                    "prn": str(prn),
                    "attempt_idx": int(attempt_idx),
                    "attempt_no": int(attempt_idx + 1),
                    "info": info,
                    "exam_student_id": int(info["exam_student_id"]),
                }
            )

    if not attempts:
        st.info("No students found.")
        return

    # 2) Picker
    if "attempt_picker" not in st.session_state:
        st.session_state["attempt_picker"] = 0

    if "jump_to_idx" in st.session_state:
        st.session_state["attempt_picker"] = st.session_state.pop("jump_to_idx")

    st.session_state["attempt_picker"] = max(
        0, min(st.session_state["attempt_picker"], len(attempts) - 1)
    )

    labels = [
        f"{a['info'].get('name','')} | PRN {a['prn']} | Attempt {a['attempt_no']} | "
        f"{a['info'].get('score',0)}/{a['info'].get('total',0)}"
        for a in attempts
    ]

    nav1, nav2, nav3 = st.columns([1, 1, 3])
    with nav1:
        if st.button("⬅ Prev", use_container_width=True):
            st.session_state["jump_to_idx"] = max(0, st.session_state["attempt_picker"] - 1)
            st.rerun()
    with nav2:
        if st.button("Next ➡", use_container_width=True):
            st.session_state["jump_to_idx"] = min(len(attempts) - 1, st.session_state["attempt_picker"] + 1)
            st.rerun()

    selected_idx = st.selectbox(
        "Choose student attempt to edit",
        options=list(range(len(labels))),
        format_func=lambda i: labels[i],
        key="attempt_picker",
    )

    # 3) Selected attempt
    chosen = attempts[selected_idx]
    prn = chosen["prn"]
    attempt_idx = chosen["attempt_idx"]
    info = chosen["info"]
    exam_student_id = chosen["exam_student_id"]

    # Reset answer widgets when switching student
    if st.session_state.get("last_exam_student_id") != exam_student_id:
        st.session_state["last_exam_student_id"] = exam_student_id
        for k in list(st.session_state.keys()):
            if k.startswith("ans_"):
                st.session_state.pop(k, None)

    st.markdown("---")
    st.subheader(
        f"{info.get('name','')} | PRN: {prn} | Attempt {chosen['attempt_no']} | "
        f"Score {info.get('score',0)}/{info.get('total',0)}"
    )

    # ---------------------------------------------------------
    # ✅ TOP: Identity + Auto-match (full width)
    # ---------------------------------------------------------

    # DB scope
    course_id = get_course_id_for_subject(subject_id)
    batch_id = get_course_batch_id(course_id) if course_id else None

    # session keys for identity widgets
    name_key = f"name_{exam_student_id}"
    prn_key = f"prn_{exam_student_id}"

    # ensure defaults exist
    if name_key not in st.session_state:
        st.session_state[name_key] = info.get("name", "")
    if prn_key not in st.session_state:
        st.session_state[prn_key] = str(prn)

    # ✅ Apply pending autofill BEFORE widgets are created
    pending_key = f"pending_autofill_{exam_student_id}"
    pending = st.session_state.pop(pending_key, None)
    if pending:
        st.session_state[name_key] = pending.get("name", "")
        st.session_state[prn_key] = pending.get("prn", "")

        # ✅ save immediately (this run, before widgets)
        final_prn = str(st.session_state.get(prn_key, "")).strip()
        final_name = str(st.session_state.get(name_key, "")).strip()
        if final_prn and final_name:
            update_exam_student_identity(exam_student_id, final_prn, final_name)
            _update_session_identity(
                subject_result=subject_result,
                old_prn=str(prn),
                entry_index=attempt_idx,
                new_prn=final_prn,
                new_name=final_name,
            )

    # ---------------------------------------------------------
    # ✅ AUTO-MATCH: if exactly 1 match, fill + save automatically
    # (no dropdown shown in that case)
    # ---------------------------------------------------------
    # compute matches BEFORE widgets
    current_prn_value_pre = str(st.session_state.get(prn_key, "") or "")
    last3_pre = _extract_last3_digits(current_prn_value_pre)

    matches: List[dict] = []
    if course_id and last3_pre:
        matches = find_students_by_last3(batch_id=batch_id, course_id=course_id, prn_last3=last3_pre)

    auto_done_key = f"auto_filled_once_{exam_student_id}"
    if not st.session_state.get(auto_done_key, False):
        if len(matches) == 1:
            m = matches[0]
            st.session_state[prn_key] = m["prn"]
            st.session_state[name_key] = m["name"]
            st.session_state[auto_done_key] = True

            # save immediately
            update_exam_student_identity(exam_student_id, m["prn"], m["name"])
            _update_session_identity(
                subject_result=subject_result,
                old_prn=str(prn),
                entry_index=attempt_idx,
                new_prn=m["prn"],
                new_name=m["name"],
            )
            st.rerun()

    # --- Top row: Name + PRN (widgets created here)
    st.markdown("#### Student identity")
    c1, c2 = st.columns([2, 2])
    with c1:
        new_name = st.text_input("Student Name", key=name_key)
    with c2:
        new_prn = st.text_input("PRN", key=prn_key)

    # --- Auto match UI
    st.markdown("### 🔎 Auto-match student (from Students table)")

    current_prn_value = str(st.session_state.get(prn_key, "") or "")
    last3 = _extract_last3_digits(current_prn_value)

    # refresh matches for display
    if not course_id:
        st.warning("Subject → Course mapping not found. Auto-match disabled.")
        matches = []
    elif not last3:
        st.warning("PRN has less than 3 digits. Auto-match needs last 3 digits.")
        matches = []
    else:
        matches = find_students_by_last3(batch_id=batch_id, course_id=course_id, prn_last3=last3)

    if len(matches) == 0:
        if course_id and last3:
            st.error(
                f"❌ Conflict: No student found in this course/batch with last3={last3}. "
                "Please type correct PRN (and name) manually below."
            )

    elif len(matches) == 1:
        # ✅ no dropdown, show info only
        st.success(f"✅ Auto-matched: {matches[0]['name']} ({matches[0]['prn']})")

    else:
        # ✅ multiple matches only -> show dropdown (NO keep current)
        st.warning(f"⚠️ {len(matches)} students share last3={last3}. Pick the correct one.")
        options = [f"{m['name']} ({m['prn']})" for m in matches]
        pick = st.selectbox("Suggested matches", options, key=f"match_pick_{exam_student_id}")

        picked_prn = pick.split("(")[-1].replace(")", "").strip()
        picked_row = next((m for m in matches if m["prn"] == picked_prn), None)
        if picked_row:
            # ✅ set pending, then rerun so it applies before widgets next time
            st.session_state[pending_key] = {
                "prn": picked_row["prn"],
                "name": picked_row["name"],
            }
            # after choosing one, mark auto done
            st.session_state[auto_done_key] = True
            st.rerun()

    st.markdown("---")

    # ---------------------------------------------------------
    # ✅ BELOW: Left Image | Right Answers + Buttons
    # ---------------------------------------------------------
    col_img, col_ans = st.columns([4, 6], gap="large")

    with col_img:
        img_path = info.get("image_path")
        if img_path and os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.info("No image available")

    with col_ans:
        with st.form(key=f"student_form_{exam_student_id}", clear_on_submit=False):

            st.markdown("#### Answers")

            details_by_q = {d["question"]: d for d in info.get("details", [])}
            updated_answers: Dict[int, str] = {}

            # 5 per row
            for row_start in range(0, len(q_numbers), 5):
                row_qs = q_numbers[row_start: row_start + 5]
                cols = st.columns(len(row_qs))
                for col, qno in zip(cols, row_qs):
                    d = details_by_q.get(qno)
                    cur_val = ""
                    if d:
                        cur_val = d.get("student_answer", "")
                        if cur_val in ("", "(blank)", "BLANK"):
                            cur_val = ""

                    with col:
                        st.caption(f"Q{qno} (key {key_map.get(qno,'')})")
                        updated_answers[qno] = (
                            st.text_input(
                                label=f"ans_{exam_student_id}_{qno}",
                                value=cur_val,
                                max_chars=1,
                                key=f"ans_{exam_student_id}_{qno}",
                                label_visibility="collapsed",
                            )
                            .strip()
                            .upper()
                        )

            st.markdown("---")
            b1, b2 = st.columns([1, 1])
            with b1:
                submit_identity = st.form_submit_button("Update Name/PRN", use_container_width=True)
            with b2:
                submit_answers = st.form_submit_button("Update Student Answers", use_container_width=True)

    # --------------------
    # After submit
    # --------------------
    if submit_identity:
        update_exam_student_identity(exam_student_id, new_prn.strip(), new_name.strip())

        _update_session_identity(
            subject_result=subject_result,
            old_prn=str(prn),
            entry_index=attempt_idx,
            new_prn=new_prn.strip(),
            new_name=new_name.strip(),
        )

        st.success("Identity updated.")
        st.rerun()

    if submit_answers:
        new_score = 0
        new_details = []

        for qno in q_numbers:
            key_ans = (key_map.get(qno) or "").strip().upper()
            ans = (updated_answers.get(qno) or "").strip().upper()

            is_blank = ans == ""
            is_correct = (not is_blank) and (ans == key_ans)
            if is_correct:
                new_score += 1

            new_details.append(
                {
                    "question": int(qno),
                    "student_answer": ans if ans else "(blank)",
                    "key_answer": key_ans,
                    "is_correct": bool(is_correct),
                    "is_blank": bool(is_blank),
                }
            )

        update_exam_student_answers(exam_student_id, new_score, new_details)

        _update_session_answers(
            subject_result=subject_result,
            prn=str(prn),
            entry_index=attempt_idx,
            new_score=new_score,
            new_details=new_details,
        )

        st.success(f"Updated score: {new_score}/{info.get('total',0)}")

        st.session_state["jump_to_idx"] = min(selected_idx + 1, len(attempts) - 1)
        st.rerun()
