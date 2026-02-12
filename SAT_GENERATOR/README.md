# SAT – Luồng sinh câu hỏi và đáp án

Dự án sinh câu hỏi SAT mới từ câu mẫu: Agent giải bài → xuất steps (hàm + ý nghĩa tham số) → LLM sinh câu hỏi mới + explanation + đáp án (và 4 choices nếu multiple-choice) → Solver tính đáp án cho câu mới. So sánh đáp án C vs D bằng LLM; nếu khớp có thể lưu câu hỏi vào thư mục `data/`.

---

## Mục lục

- [Sơ đồ luồng](#sơ-đồ-luồng-flowmd)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Yêu cầu & Cài đặt](#yêu-cầu--cài-đặt)
- [Cấu hình](#cấu-hình)
- [Chạy luồng (run_flow.py)](#chạy-luồng-đầy-đủ-run_flowpy)
- [Kết quả trả về](#kết-quả-trả-về)
- [Web Demo](#web-demo)
- [Các module chính](#các-module-chính)

---

## Sơ đồ luồng (flow.md)

```
A (question / explanation / correct_answer)
   ├─→ B: Agent sinh steps_function_and_meaning.json
   ├─→ C: Gen câu hỏi mới, explanation và đáp án (4 choices nếu multiple-choice)
   └─→ D: Sinh đáp án cho câu hỏi mới (dựa vào file JSON từ B và câu hỏi từ C)
```

- **A**: Đầu vào là một câu mẫu (question, explanation, correct_answer, choices nếu có) từ `questions_practice_test.json`.
- **B**: Agent (LangGraph) giải bài → tạo file `steps_function_and_meaning.json` (tên hàm + ý nghĩa tham số từng bước).
- **C**: Module sinh câu hỏi mới, **explanation** và **đáp án** (chỉ đổi số, giữ format HTML/MathML). Nếu là **multiple-choice** thì sinh đúng **4 lựa chọn** (A, B, C, D) và **1 đáp án đúng** (chữ cái).
- **D**: Dùng file steps từ B và câu hỏi mới từ C → `sat_math_solver` tính đáp án cho câu mới.

---

## Cấu trúc thư mục

| Thư mục / File | Mô tả |
|----------------|--------|
| `agent.py` | Agent LangGraph giải bài toán, xuất steps (hàm + param meaning). |
| `generate_question_langchain.py` | Sinh câu hỏi mới + explanation + đáp án (và 4 choices nếu multiple-choice). |
| `sat_math_solver.py` | Solver: dùng steps JSON + câu hỏi mới → gọi tools theo thứ tự, trả về đáp án. |
| `run_flow.py` | Chạy toàn bộ luồng A → B → C → D (CLI). |
| `app.py` | Web demo Flask: xem câu gốc, chạy flow, so sánh C vs D (LLM), Save sample vào `data/`. |
| `templates/demo.html` | Giao diện web demo. |
| `tools.py` | Các math tools (add, multiply, …) dùng bởi Agent và Solver. |
| `mathml_parser.py` | Parse MathML sang text đọc được. |
| `questions_practice_test.json` | Danh sách câu mẫu (đầu vào). |
| `steps_function_and_meaning.json` | File steps do bước B tạo (đầu vào cho D). |
| `data/` | Thư mục lưu câu hỏi đã sinh khi bấm **Save sample** (file `generated_YYYYMMDD_HHMMSS.json`). |
| `flow.md` | Sơ đồ luồng (Mermaid). |

---

## Yêu cầu & Cài đặt

- **Python 3.10+**
- [uv](https://github.com/astral-sh/uv) (hoặc pip)
- **OpenAI API key**
- Các thư viện trong `requirements.txt`: `langchain-openai`, `langgraph`, `pydantic`, `python-dotenv`, `flask`, …

### Cài đặt (dùng uv)

```bash
cd /path/to/SAT
uv venv
source .venv/bin/activate   # Linux/macOS; Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

Hoặc dùng pip:

```bash
pip install -r requirements.txt
```

---

## Cấu hình

Tạo file `.env` trong thư mục gốc (cùng cấp với `run_flow.py`, `app.py`):

```env
OPENAI_API_KEY=sk-...
```

Hoặc export trước khi chạy:

```bash
export OPENAI_API_KEY=sk-...
```

---

## Chạy luồng đầy đủ (run_flow.py)

Chạy một lần toàn bộ luồng A → B → C → D từ dòng lệnh:

```bash
python run_flow.py
```

### Tham số dòng lệnh

| Tham số | Mô tả | Mặc định |
|--------|--------|----------|
| `--sample-index` | Index câu mẫu trong file questions (0, 1, 2, …) | `0` |
| `--question-id` | Lấy câu mẫu theo id thay vì index | — |
| `--questions-path` | Đường dẫn file danh sách câu hỏi | `questions_practice_test.json` |
| `--out-dir` | Thư mục ghi file steps JSON và kết quả | Thư mục hiện tại |
| `--steps-json` | Tên file steps JSON (ghi trong `--out-dir`) | `steps_function_and_meaning.json` |
| `--model` | Model LLM (OpenAI) | `gpt-4o-mini` |
| `--quiet` | Bật chế độ ít log | — |
| `--save-result` | Đường dẫn file JSON để lưu toàn bộ kết quả flow | — |

### Ví dụ

```bash
# Chạy với câu mẫu mặc định (index 0)
python run_flow.py

# Chạy với câu mẫu thứ 2
python run_flow.py --sample-index 2

# Chạy với câu mẫu theo id
python run_flow.py --question-id "06868720-2332-4e01-b188-4f2342bc60a9" --save-result flow_result.json

# Ghi file steps và làm việc trong thư mục output
python run_flow.py --out-dir ./output

# Lưu toàn bộ kết quả (new_question_item, answer_result, …) ra file
python run_flow.py --save-result flow_result.json

# Chạy ít log + lưu kết quả
python run_flow.py --quiet --save-result flow_result.json
```

---

## Kết quả trả về

Sau khi chạy, `run_flow` trả về (và có thể lưu qua `--save-result`) một dict gồm:

| Trường | Mô tả |
|--------|--------|
| **steps_json_path** | Đường dẫn file `steps_function_and_meaning.json` đã ghi ở bước B. |
| **new_question_item** | Dict câu hỏi mới (bước C): **question**, **explanation**, **correct_answer**; nếu multiple-choice thì thêm **choices** (đúng 4 phần tử A, B, C, D). Cùng cấu trúc với item trong `questions_practice_test.json`. |
| **new_question_text** | Nội dung câu hỏi mới (chuỗi HTML/MathML). |
| **answer_result** | Kết quả từ `sat_math_solver` (bước D): `final_result`, `steps_detail`, `error`. |
| **error** | Lỗi tổng của cả luồng (nếu có). |

---

## Web Demo

Chạy web demo: nhập **Question ID** → xem câu gốc → chạy flow → xem câu hỏi mới, explanation, đáp án (C) và kết quả solver (D). So sánh đáp án C vs D bằng LLM; nếu khớp thì hiện nút **Save sample** để lưu câu hỏi vào `data/`.

```bash
python app.py
```

Mở trình duyệt: **http://localhost:5000**

### Các bước sử dụng

1. **Xem câu gốc**: Nhập Question ID (vd: `06868720-2332-4e01-b188-4f2342bc60a9`) → bấm **Xem câu gốc**. Hiển thị câu hỏi, đáp án đúng, explanation từ `questions_practice_test.json`.
2. **Chạy flow**: Bấm **Chạy flow** → chạy luồng A → B → C → D. Kết quả hiển thị:
   - Câu hỏi mới (và 4 đáp án nếu multiple-choice),
   - Explanation (gen từ C),
   - Đáp án từ C (correct_answer),
   - Đáp án từ D (final_result + steps).
3. **So sánh đáp án C vs D**: Backend dùng **LLM** để so sánh đáp án từ C (generated) và từ D (solver). Nếu LLM trả lời hai đáp án tương đương → `answers_match: true`.
4. **Save sample**: Chỉ khi **đáp án C và D khớp** (`answers_match: true`) thì mới hiện nút **Save sample**. Bấm **Save sample** → câu hỏi (question + explanation + choices + correct_answer) được lưu vào file JSON trong thư mục **`data/`** (tên file: `generated_YYYYMMDD_HHMMSS.json`).

### API (app.py)

| Endpoint | Mô tả |
|----------|--------|
| `GET /` | Trang demo. |
| `GET /api/question/<question_id>` | Lấy câu gốc theo ID. |
| `POST /api/run-flow` | Body: `{ "question_id": "..." }`. Chạy flow, trả về `new_question_item`, `answer_result`, **answers_match** (so sánh C vs D bằng LLM), … |
| `POST /api/save-question` | Body: `{ "new_question_item": { ... } }`. Lưu câu hỏi vào `data/generated_YYYYMMDD_HHMMSS.json`. |

---

## Các module chính

- **agent.py**: LangGraph math agent — giải bài theo explanation, gọi tools (add, multiply, …), mỗi bước có node giải thích ý nghĩa tham số; xuất `ExecutionTrace` và export steps ra JSON.
- **generate_question_langchain.py**: LLM sinh câu hỏi mới + explanation + đáp án từ câu mẫu (chỉ đổi số, giữ format). Multiple-choice: sinh đúng 4 choices + correct_answer_letter (A/B/C/D). Hàm `load_sample_question` dùng chung cho run_flow và app.
- **sat_math_solver.py**: Đọc `steps_function_and_meaning.json`, với câu hỏi mới gọi từng step (tool + param meaning), LLM chọn giá trị tham số → trả về `final_result`.
- **app.py**: Flask app — trang demo, API run-flow, so sánh C vs D bằng LLM (`_answers_match_llm`), API save-question ghi file vào `data/`.
- **tools.py**: Định nghĩa các math tools (add, multiply, divide, …) dùng bởi Agent và Solver.

---

## Thư mục data

Thư mục **`data/`** chứa các câu hỏi đã sinh khi người dùng bấm **Save sample** trên Web Demo (chỉ hiện khi đáp án C và D khớp). Mỗi file có dạng `generated_YYYYMMDD_HHMMSS.json`, cấu trúc giống một item trong `questions_practice_test.json` (id, subject, section, category, question với question, explanation, choices, correct_answer, …).
