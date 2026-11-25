from train.teacher_rows import select_teacher_rows


def grade(answer, reference, gold):
    return {"ok": answer.startswith("OK"), "rouge": float(len(answer))}


def test_best_good_answer_per_question_across_runs_and_bad_ones_dropped():
    run1 = {1: "OK short", 2: "bad", 3: "OK a"}
    run2 = {1: "OK a longer one", 2: "bad too"}
    rows = select_teacher_rows([run1, run2], lambda q: "ref", lambda q: {"1"}, grade, exclude_copy=False)
    assert {r["qid"]: r["answer"] for r in rows} == {1: "OK a longer one", 3: "OK a"}


def test_an_answer_that_copies_the_reference_is_dropped():
    rows = select_teacher_rows([{1: "OK same"}], lambda q: "OK same", lambda q: {"1"}, grade)
    assert rows == []
