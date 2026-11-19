# Findings — non-thinking readers (agent, 2026-11-18; HF raw files)

| model | temp | top_p | top_k | repetition | source |
|---|---|---|---|---|---|
| AITeamVN/Vi-Qwen2-3B-RAG | 0.7 config; card example 0.1 | 0.8 | 20 | 1.05 | generation_config.json, README |
| Qwen2.5-3B-Instruct (~3.1B, unverified) | 0.7 | 0.8 | 20 | 1.05 | generation_config.json |
| Llama-3.2-3B-Instruct (~3.2B) | 0.6 | 0.9 | — | — | unsloth mirror (official gated); Vietnamese not official (unverified) |

Vi-Qwen2-3B-RAG card (a copy of the 7B card): trained for noise robustness, negative rejection, information
integration; max_new_tokens 2048, context 8192; no 3B benchmark numbers. Prompt: system "Bạn là một trợ lí Tiếng Việt
nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể." user "Chú ý các yêu cầu sau: - Câu trả lời
phải chính xác và đầy đủ nếu ngữ cảnh có câu trả lời. - Chỉ sử dụng các thông tin có trong ngữ cảnh được cung cấp.
- Chỉ cần từ chối trả lời và không suy luận gì thêm nếu ngữ cảnh không có câu trả lời. Hãy trả lời câu hỏi dựa trên
ngữ cảnh: ### Ngữ cảnh : {context} ### Câu hỏi : {question} ### Trả lời :" (contexts as "- Ngữ cảnh 1:" …).
No Vietnamese legal RAG report on greedy vs sampling found.
Sources: https://huggingface.co/AITeamVN/Vi-Qwen2-3B-RAG · https://huggingface.co/Qwen/Qwen2.5-3B-Instruct ·
https://huggingface.co/unsloth/Llama-3.2-3B-Instruct
