flowchart TD
A[question / explanation / correct_answer]


A --> B[Agent sinh steps_function_and_meaning.json]
A --> C[Gen câu hỏi mới, explanation và đáp án]


B --> D[Sinh đáp án cho câu hỏi mới\n(dựa vào file JSON)]
C --> D