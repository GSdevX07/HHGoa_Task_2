"""
Hugging Face Spaces Entrypoint with ZeroGPU and Gradio 6 Support
===============================================================
"""

import os
import sys
import asyncio
import gradio as gr

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ZeroGPU decorator compatibility
try:
    import spaces
except ImportError:
    spaces = None

from backend.main import (
    app as fastapi_app,
    _execute_rag_pipeline,
    stt_engine,
    model_harness,
    dense_retriever,
    bm25_retriever,
)


def _core_rag_logic(audio_path, text_query, lang_code, provider, strategy):
    """Core RAG logic callable within GPU or CPU context."""
    async def _async_run():
        transcript = ""
        stt_latency = 0.0
        provider_used = provider

        # If audio provided, transcribe
        if audio_path and os.path.exists(audio_path):
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
            stt_res = await stt_engine.transcribe_audio(
                audio_bytes=audio_bytes,
                filename=os.path.basename(audio_path),
                language_code=lang_code,
                provider_override=provider,
            )
            if stt_res.get("success"):
                transcript = stt_res.get("transcript", "")
                stt_latency = stt_res.get("latency_ms", 0.0)
                provider_used = stt_res.get("provider", provider)
            elif text_query and text_query.strip():
                transcript = text_query.strip()
                provider_used = "text_fallback"
            else:
                return (
                    f"⚠️ STT Transcription Error: {stt_res.get('error', 'Unknown')}\n\nTip: Add SARVAM_API_KEY in Space Settings.",
                    "0.0",
                    "N/A",
                    "",
                )
        elif text_query and text_query.strip():
            transcript = text_query.strip()
            provider_used = "text_input"
        else:
            return "Please provide an audio recording or type a text query.", "0.0", "N/A", ""

        # Execute full RAG pipeline
        rag_res = await _execute_rag_pipeline(
            transcript=transcript,
            stt_latency_ms=stt_latency,
            stt_provider=provider_used,
            chunking_strategy=strategy,
            language_code=lang_code,
            enable_guardrails=True,
        )

        res_dict = rag_res.model_dump()
        answer = res_dict.get("answer", "")
        groundedness = f"{res_dict.get('groundedness_score', 0.0):.2f}"
        language = res_dict.get("detected_language", lang_code)

        # Citations formatting
        cits = res_dict.get("citations", [])
        cit_text = ""
        for i, c in enumerate(cits, 1):
            cit_text += f"**[{i}] {c.get('chunk_id')}** (score: {c.get('similarity_score', 0):.3f})\n> {c.get('snippet', '')}\n\n"

        latency_breakdown = (
            f"⚡ **Latency Breakdown (Total: {res_dict.get('total_latency_ms', 0):.1f} ms)**\n"
            f"- STT: {res_dict.get('stt_latency_ms', 0):.1f} ms\n"
            f"- Retrieval: {res_dict.get('retrieval_latency_ms', 0):.1f} ms\n"
            f"- Guardrails: {res_dict.get('guardrail_latency_ms', 0):.1f} ms\n"
            f"- LLM Generation: {res_dict.get('llm_latency_ms', 0):.1f} ms"
        )

        full_output = f"### 💬 Answer:\n{answer}\n\n---\n### 📚 Citations:\n{cit_text or 'No citations'}\n\n---\n{latency_breakdown}"
        return full_output, groundedness, language, transcript

    return asyncio.run(_async_run())


# Decorate with spaces.GPU if ZeroGPU is active
if spaces is not None:
    @spaces.GPU(duration=60)
    def process_rag_query(audio_path, text_query, lang_code, provider, strategy):
        return _core_rag_logic(audio_path, text_query, lang_code, provider, strategy)
else:
    def process_rag_query(audio_path, text_query, lang_code, provider, strategy):
        return _core_rag_logic(audio_path, text_query, lang_code, provider, strategy)


# ── Gradio UI Construction ───────────────────────────────────────────────────

custom_css = """
.gradio-container { font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif; }
.main-header { text-align: center; margin-bottom: 20px; }
.main-header h1 { font-size: 2.2rem; font-weight: 800; color: #1e1e2f; }
"""

with gr.Blocks(title="🎙️ Voice RAG — MSMARCO-XI") as demo:
    gr.Markdown(
        """
        # 🎙️ Multilingual Voice-Enabled RAG System
        ### AI4Bharat MSMARCO-XI • Sub-200ms Latency Engine • Sarvam AI & Groq LLM
        """
    )

    with gr.Tabs():
        with gr.TabItem("🎙️ Voice & Text RAG Query"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="🎙️ Speak or Upload Audio (Indic / English)",
                    )
                    text_input = gr.Textbox(
                        label="✍️ Or Enter Text Query",
                        placeholder="e.g. What is the capital of India? / भारत की राजधानी क्या है?",
                        lines=2,
                    )
                    with gr.Row():
                        lang_select = gr.Dropdown(
                            label="Language",
                            choices=[
                                ("Hindi (हिन्दी)", "hi-IN"),
                                ("English", "en-US"),
                                ("Telugu (తెలుగు)", "te-IN"),
                                ("Tamil (தமிழ்)", "ta-IN"),
                                ("Bengali (বাংলা)", "bn-IN"),
                                ("Marathi (मराठी)", "mr-IN"),
                                ("Gujarati (ગુજરાતી)", "gu-IN"),
                                ("Kannada (ಕನ್ನಡ)", "kn-IN"),
                                ("Malayalam (മലയാളം)", "ml-IN"),
                            ],
                            value="hi-IN",
                        )
                        stt_select = gr.Dropdown(
                            label="STT Provider",
                            choices=[("Sarvam AI (Saarika)", "sarvam"), ("ElevenLabs Scribe", "elevenlabs")],
                            value="sarvam",
                        )
                        chunk_select = gr.Dropdown(
                            label="Chunking Strategy",
                            choices=[("Semantic (Dynamic)", "semantic"), ("Fixed (256 words)", "fixed"), ("Hierarchical", "hierarchical")],
                            value="semantic",
                        )

                    submit_btn = gr.Button("🚀 Run Voice RAG", variant="primary", size="lg")

                with gr.Column(scale=1):
                    transcript_out = gr.Textbox(label="📝 Transcribed Query", interactive=False)
                    with gr.Row():
                        groundedness_out = gr.Textbox(label="🛡️ Groundedness Score", interactive=False)
                        lang_out = gr.Textbox(label="🌐 Detected Language", interactive=False)
                    output_markdown = gr.Markdown(label="Response")

            submit_btn.click(
                fn=process_rag_query,
                inputs=[audio_input, text_input, lang_select, stt_select, chunk_select],
                outputs=[output_markdown, groundedness_out, lang_out, transcript_out],
            )

        with gr.TabItem("🧩 System Diagnostics & Info"):
            gr.Markdown(
                """
                ### 📊 System Status
                - **Dataset**: AI4Bharat MSMARCO-XI (Multilingual)
                - **Dense Retrieval**: `all-MiniLM-L6-v2` (384-dim FAISS)
                - **Sparse Retrieval**: BM25 (Rank-BM25)
                - **Reranker**: `BAAI/bge-reranker-v2-m3`
                - **LLM Synthesis**: Groq Llama-3.3-70B / Extractive Synthesizer
                - **Hardware**: Accelerated via Hugging Face ZeroGPU
                """
            )


# Always launch Gradio demo directly for Hugging Face Spaces & ZeroGPU
demo.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
    css=custom_css,
    theme=gr.themes.Soft(),
)
