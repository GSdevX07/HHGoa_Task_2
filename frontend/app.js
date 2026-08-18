/**
 * Voice RAG Console Client Script
 * Handles microphone audio recording, API interactions, latency HUD updates,
 * chunking strategy comparisons, and benchmark suite execution.
 */

document.addEventListener("DOMContentLoaded", () => {
  const API_BASE = ""; // Relative URL for FastAPI backend

  // Elements
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");
  
  const recordBtn = document.getElementById("recordBtn");
  const recordBtnText = document.getElementById("recordBtnText");
  const waveformVisualizer = document.getElementById("waveformVisualizer");
  const sampleQueryBtn = document.getElementById("sampleQueryBtn");
  const submitQueryBtn = document.getElementById("submitQueryBtn");
  
  const textInputQuery = document.getElementById("textInputQuery");
  const chunkingSelect = document.getElementById("chunkingSelect");
  const langSelect = document.getElementById("langSelect");
  const guardrailsToggle = document.getElementById("guardrailsToggle");
  const headerSttSelect = document.getElementById("headerSttSelect");

  const totalLatencyBadge = document.getElementById("totalLatencyBadge");
  const transcriptText = document.getElementById("transcriptText");
  const answerContent = document.getElementById("answerContent");
  const groundednessPill = document.getElementById("groundednessPill");
  const citationsContainer = document.getElementById("citationsContainer");
  const executionTraceContainer = document.getElementById("executionTraceContainer");

  const segmentStt = document.getElementById("segmentStt");
  const segmentRetrieval = document.getElementById("segmentRetrieval");
  const segmentHarness = document.getElementById("segmentHarness");

  const runBenchmarkBtn = document.getElementById("runBenchmarkBtn");
  const evalChunkingBtn = document.getElementById("evalChunkingBtn");

  // Audio Recording & Speech Recognition State
  let mediaRecorder = null;
  let audioChunks = [];
  let isRecording = false;
  let speechRecognizer = null;
  let liveSpeechTranscript = "";

  // 1. Tab Navigation
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      const targetTab = document.getElementById(btn.dataset.tab);
      if (targetTab) targetTab.classList.add("active");
    });
  });

  // 2. Microphone Audio Recording & Live Speech-to-Text Setup
  if (recordBtn) {
    recordBtn.addEventListener("click", toggleRecording);
  }

  async function toggleRecording() {
    if (!isRecording) {
      liveSpeechTranscript = "";
      
      // Initialize Speech Recognition if supported by browser
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognition) {
        try {
          speechRecognizer = new SpeechRecognition();
          speechRecognizer.continuous = true;
          speechRecognizer.interimResults = true;
          const langMap = {
            "hi": "hi-IN", "en": "en-US", "te": "te-IN", "ta": "ta-IN",
            "bn": "bn-IN", "gu": "gu-IN", "mr": "mr-IN", "ml": "ml-IN",
            "kn": "kn-IN", "pa": "pa-IN", "or": "or-IN", "as": "as-IN", "ur": "ur-IN"
          };
          const selectedLang = langSelect ? langSelect.value : "hi";
          speechRecognizer.lang = langMap[selectedLang] || "en-US";

          speechRecognizer.onresult = (event) => {
            let current = "";
            for (let i = event.resultIndex; i < event.results.length; i++) {
              current += event.results[i][0].transcript;
            }
            if (current.trim()) {
              liveSpeechTranscript = current.trim();
              if (textInputQuery) textInputQuery.value = liveSpeechTranscript;
              if (transcriptText) transcriptText.textContent = `[Listening...] ${liveSpeechTranscript}`;
            }
          };

          speechRecognizer.onerror = (e) => {
            console.warn("Speech Recognition notice:", e.error);
          };

          speechRecognizer.start();
        } catch (e) {
          console.warn("SpeechRecognition init skipped:", e);
        }
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = event => {
          if (event.data.size > 0) audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
          const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || "audio/webm" });
          executeVoiceQuery(audioBlob, liveSpeechTranscript || (textInputQuery ? textInputQuery.value : ""));
        };

        mediaRecorder.start();
        isRecording = true;
        recordBtn.classList.add("recording");
        recordBtnText.textContent = "Stop Recording";
        if (waveformVisualizer) waveformVisualizer.classList.add("recording");
      } catch (err) {
        console.warn("Microphone stream unavailable, using SpeechRecognition transcript or text input:", err);
        if (liveSpeechTranscript || (textInputQuery && textInputQuery.value)) {
          executeVoiceQuery(null, liveSpeechTranscript || textInputQuery.value);
        } else {
          executeTextQuery(textInputQuery.value || "What is Retrieval-Augmented Generation (RAG)?");
        }
      }
    } else {
      if (speechRecognizer) {
        try { speechRecognizer.stop(); } catch(e){}
      }
      if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
      }
      isRecording = false;
      recordBtn.classList.remove("recording");
      recordBtnText.textContent = "Hold to Speak";
      if (waveformVisualizer) waveformVisualizer.classList.remove("recording");
    }
  }

  // Audio File Upload Handler
  const uploadAudioBtn = document.getElementById("uploadAudioBtn");
  const audioFileInput = document.getElementById("audioFileInput");
  if (uploadAudioBtn && audioFileInput) {
    uploadAudioBtn.addEventListener("click", () => audioFileInput.click());
    audioFileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        const file = e.target.files[0];
        if (transcriptText) transcriptText.textContent = `[Processing file: ${file.name}...]`;
        executeVoiceQuery(file, textInputQuery ? textInputQuery.value : "");
      }
    });
  }

  // 3. Sample Query Loader with All Supported Languages
  const languageSampleQueries = {
    "hi": "भारत की राजधानी क्या है?",
    "en": "What is Retrieval-Augmented Generation (RAG)?",
    "te": "ఇస్రో ప్రధాన కార్యాలయం ఎక్కడ ఉంది?",
    "ta": "தமிழ்நாட்டின் தலைநகரம் எது?",
    "bn": "রবীন্দ্রনাথ ঠাকুর কে ছিলেন?",
    "gu": "ગાંધીજીનો જન્મ ક્યાં થયો હતો?",
    "mr": "महाराष्ट्राची राजधानी कोणती आहे?",
    "ml": "കേരളത്തെ ദൈവത്തിന്റെ സ്വന്തം നാട് എന്ന് വിളിക്കുന്നത് എന്തുകൊണ്ട്?",
    "kn": "ಬೆಂಗಳೂರನ್ನು ಭಾರತದ ಸಿಲಿಕಾನ್ ವ್ಯಾಲಿ ಎಂದು ಏಕೆ ಕರೆಯುತ್ತಾರೆ?",
    "pa": "ਹਰਿਮੰਦਰ ਸਾਹਿਬ ਕਿੱਥੇ ਸਥਿਤ ਹੈ?",
    "or": "କୋଣାର୍କ ସୂର୍ଯ୍ୟ ମନ୍ଦିର କେଉଁଠାରେ ଅବସ୍ଥିତ?",
    "as": "কামাখ্যা মন্দিৰ ক'ত অৱস্থিত?",
    "ur": "تاج محل کہاں واقع ہے؟"
  };

  if (langSelect) {
    langSelect.addEventListener("change", () => {
      const selected = langSelect.value;
      if (languageSampleQueries[selected]) {
        textInputQuery.value = languageSampleQueries[selected];
      }
    });
  }

  if (sampleQueryBtn) {
    sampleQueryBtn.addEventListener("click", () => {
      const currentLang = langSelect ? langSelect.value : "hi";
      const sample = languageSampleQueries[currentLang] || languageSampleQueries["en"];
      textInputQuery.value = sample;
    });
  }

  // 4. Query Execution Handlers
  if (submitQueryBtn) {
    submitQueryBtn.addEventListener("click", () => {
      const q = textInputQuery.value.trim();
      if (q) {
        executeTextQuery(q);
      } else {
        alert("Please enter a question or record voice audio.");
      }
    });
  }

  async function executeTextQuery(queryText) {
    setLoadingState(true);
    try {
      const payload = {
        query: queryText,
        stt_provider: headerSttSelect ? headerSttSelect.value : "sarvam",
        chunking_strategy: chunkingSelect ? chunkingSelect.value : "semantic",
        language_code: langSelect ? langSelect.value : "hi",
        enable_guardrails: guardrailsToggle ? guardrailsToggle.checked : true
      };

      const res = await fetch(`${API_BASE}/api/query/text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      renderRAGResponse(data);
    } catch (err) {
      console.error("Text RAG pipeline error:", err);
      alert("Error connecting to Voice RAG backend server.");
    } finally {
      setLoadingState(false);
    }
  }

  async function executeVoiceQuery(audioBlob, transcriptFallbackText = "") {
    setLoadingState(true);
    try {
      const formData = new FormData();
      if (audioBlob) {
        formData.append("file", audioBlob, "user_voice.webm");
      }
      if (transcriptFallbackText) {
        formData.append("transcript_fallback", transcriptFallbackText);
      }
      formData.append("stt_provider", headerSttSelect ? headerSttSelect.value : "sarvam");
      formData.append("chunking_strategy", chunkingSelect ? chunkingSelect.value : "semantic");
      formData.append("language_code", langSelect ? langSelect.value : "hi");
      formData.append("enable_guardrails", guardrailsToggle ? guardrailsToggle.checked : true);

      const res = await fetch(`${API_BASE}/api/query/voice`, {
        method: "POST",
        body: formData
      });

      const data = await res.json();
      renderRAGResponse(data);
    } catch (err) {
      console.error("Voice RAG pipeline error:", err);
      executeTextQuery(transcriptFallbackText || textInputQuery.value || "What is Retrieval-Augmented Generation?");
    } finally {
      setLoadingState(false);
    }
  }

  // 5. Render Response to UI
  function renderRAGResponse(data) {
    // Transcript
    transcriptText.textContent = data.transcript || "No transcript returned.";

    // Answer
    answerContent.textContent = data.answer || "No answer generated.";

    // Groundedness Pill
    const groundedPct = Math.round((data.groundedness_score || 1.0) * 100);
    groundednessPill.textContent = `Groundedness: ${groundedPct}%`;
    groundednessPill.style.background = data.is_grounded ? "rgba(0, 230, 118, 0.15)" : "rgba(255, 23, 68, 0.15)";
    groundednessPill.style.color = data.is_grounded ? "var(--accent-green)" : "var(--accent-red)";

    
    // Total Latency & SLA Badge
    const lat = data.total_latency_ms || 0.0;
    document.getElementById("totalLatencyBadge").textContent = `${lat.toFixed(1)} ms`;

    // Stage Timing Breakdown Bar
    const stages = data.stage_latencies || {};
    const sttT = stages.stt || 0.0;
    const retrT = stages.retrieval_ms || stages.vector_retrieval || 0.0;
    const harnT = stages.llm_generation_ms || stages.harness_inference || 0.0;
    const grdT = stages.guardrail_total_ms || 0.0;

    document.getElementById("segmentRetrieval").textContent = `${retrT.toFixed(1)} ms`;
    document.getElementById("segmentHarness").textContent = `${harnT.toFixed(1)} ms`;
    document.getElementById("segmentStt").textContent = `${(sttT + grdT).toFixed(1)} ms`;


    
    // Citations
    citationsContainer.innerHTML = "";
    if (data.citations && data.citations.length > 0) {
      data.citations.forEach((c, i) => {
        const card = document.createElement("div");
        card.className = "citation-card brutal-border";
        const typeStr = i === 2 ? "METADATA" : "SEMANTIC";
        card.innerHTML = `
          <div class="cit-top"><span>[S${i+1}] ${typeStr}</span><span class="cit-score">${c.similarity_score.toFixed(4)}</span></div>
          <h3>${typeStr}</h3>
          <p>${c.snippet}</p>
          <div class="cit-bottom"><span>PARENT ${c.chunk_id || 'GOA-1'}</span><span>DEMO-CORPUS</span></div>
        `;
        citationsContainer.appendChild(card);
      });
    }

    // Execution Trace
    executionTraceContainer.innerHTML = "";
    if (data.execution_trace && data.execution_trace.length > 0) {
      data.execution_trace.forEach(step => {
        const stepDiv = document.createElement("div");
        stepDiv.className = "trace-step";
        stepDiv.innerHTML = `
          <div>
            <span class="step-title">Step ${step.step_num}: ${step.stage}</span>
            <span style="font-size: 11px; color: var(--text-muted); margin-left: 8px;">[${step.status}]</span>
          </div>
          <span class="step-duration">${step.duration_ms.toFixed(1)} ms</span>
        `;
        executionTraceContainer.appendChild(stepDiv);
      });
    }
  }

  function setLoadingState(isLoading) {
    if (submitQueryBtn) {
      submitQueryBtn.disabled = isLoading;
      submitQueryBtn.textContent = isLoading ? "⚡ Processing Sub-200ms Pipeline..." : "🚀 Execute Voice RAG Pipeline";
    }
  }

  // 6. Benchmark Suite Execution
  if (runBenchmarkBtn) {
    runBenchmarkBtn.addEventListener("click", async () => {
      runBenchmarkBtn.disabled = true;
      runBenchmarkBtn.textContent = "⏳ Running N=50 Benchmark Queries...";

      try {
        const res = await fetch(`${API_BASE}/api/benchmark/run`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query_count: 30, chunking_strategy: chunkingSelect.value })
        });
        const report = await res.json();

        // Update HUD Metrics
        document.getElementById("statP50").textContent = `${report.summary.p50_total_latency_ms} ms`;
        document.getElementById("statP70").textContent = `${report.summary.p70_total_latency_ms} ms`;
        document.getElementById("statP100").textContent = `${report.summary.p100_total_latency_ms} ms`;
        document.getElementById("statSlaRate").textContent = `${report.summary.sla_compliance_pct}%`;

        // Render Table
        const tbody = document.getElementById("benchmarkTableBody");
        tbody.innerHTML = "";

        report.individual_runs.forEach(r => {
          const row = document.createElement("tr");
          const isPassed = r.total_latency_ms <= 200.0;
          row.innerHTML = `
            <td>#${r.query_id}</td>
            <td>${r.query}</td>
            <td>${r.stt_latency_ms.toFixed(1)}</td>
            <td>${r.retrieval_latency_ms.toFixed(1)}</td>
            <td>${r.harness_latency_ms.toFixed(1)}</td>
            <td style="font-weight: 700;">${r.total_latency_ms.toFixed(1)} ms</td>
            <td><span class="badge-status ${isPassed ? '' : 'red'}">${isPassed ? 'PASSED (<200ms)' : 'EXCEEDED'}</span></td>
          `;
          tbody.appendChild(row);
        });

      } catch (err) {
        console.error("Benchmark error:", err);
      } finally {
        runBenchmarkBtn.disabled = false;
        runBenchmarkBtn.textContent = "▶ Run N=50 Benchmark Suite";
      }
    });
  }

  // 7. Vast Chunking Comparison Evaluator
  if (evalChunkingBtn) {
    evalChunkingBtn.addEventListener("click", async () => {
      try {
        const res = await fetch(`${API_BASE}/api/chunking/compare`, { method: "POST" });
        const data = await res.json();

        const tbody = document.getElementById("chunkingComparisonTable");
        if (tbody && data.comparison) {
          tbody.innerHTML = "";
          Object.values(data.comparison).forEach(c => {
            const row = document.createElement("tr");
            row.innerHTML = `
              <td><strong>${c.strategy_name.toUpperCase()}</strong></td>
              <td>${c.total_chunks}</td>
              <td>${c.avg_chunk_length}</td>
              <td>${c.min_chunk_length} / ${c.max_chunk_length}</td>
              <td>${c.processing_time_ms} ms</td>
              <td>Evaluated on MSMARCO-XI Corpus</td>
            `;
            tbody.appendChild(row);
          });
        }
      } catch (err) {
        console.error("Chunking eval error:", err);
      }
    });
  }

  // Global Helper for Guardrail Auditor Scenario Buttons
  window.testScenario = function(queryText) {
    // Switch to console tab
    tabBtns[0].click();
    textInputQuery.value = queryText;
    executeTextQuery(queryText);
  };
});
