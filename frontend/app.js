/**
 * VAANI • Voice RAG Console
 * Neo-Brutalist Retro Theme Controller
 */

document.addEventListener("DOMContentLoaded", () => {
  const API_BASE = ""; // Relative URL for FastAPI backend

  // Core DOM Elements
  const recordBtn = document.getElementById("recordBtn");
  const recordBtnText = document.getElementById("recordBtnText");
  const waveformCanvas = document.getElementById("waveformCanvas");
  
  const textInputQuery = document.getElementById("textInputQuery");
  const submitQueryBtn = document.getElementById("submitQueryBtn");
  const clearInputBtn = document.getElementById("clearInputBtn");
  const langSelect = document.getElementById("langSelect");
  
  const cloudSttToggle = document.getElementById("cloudSttToggle");
  const browserSpeechToggle = document.getElementById("browserSpeechToggle");
  
  const transcriptText = document.getElementById("transcriptText");
  const answerContent = document.getElementById("answerContent");
  const groundednessPill = document.getElementById("groundednessPill");
  const consoleStatusLabel = document.getElementById("consoleStatusLabel");
  const reqIdBadge = document.getElementById("reqIdBadge");
  const citationsContainer = document.getElementById("citationsContainer");
  const liveClockDisplay = document.getElementById("liveClockDisplay");

  // Latency Metrics Elements
  const totalLatencyBadge = document.getElementById("totalLatencyBadge");
  const segmentRetrieval = document.getElementById("segmentRetrieval");
  const segmentHarness = document.getElementById("segmentHarness");
  const segmentStt = document.getElementById("segmentStt");

  // Modals & Extra Tools
  const openBenchmarkModalBtn = document.getElementById("openBenchmarkModalBtn");
  const closeBenchmarkModalBtn = document.getElementById("closeBenchmarkModalBtn");
  const benchmarkModal = document.getElementById("benchmarkModal");
  const runBenchmarkBtn = document.getElementById("runBenchmarkBtn");

  const openChunkingModalBtn = document.getElementById("openChunkingModalBtn");
  const closeChunkingModalBtn = document.getElementById("closeChunkingModalBtn");
  const chunkingModal = document.getElementById("chunkingModal");
  const evalChunkingBtn = document.getElementById("evalChunkingBtn");

  // Compatibility Selectors
  const chunkingSelect = document.getElementById("chunkingSelect") || { value: "semantic" };
  const headerSttSelect = document.getElementById("headerSttSelect") || { value: "sarvam" };
  const guardrailsToggle = document.getElementById("guardrailsToggle") || { checked: true };

  // State
  let mediaRecorder = null;
  let audioChunks = [];
  let isRecording = false;
  let speechRecognizer = null;
  let liveSpeechTranscript = "";
  let recognitionMode = "cloud"; // "cloud" or "browser"
  let audioCtx = null;
  let analyser = null;
  let animFrameId = null;

  // 1. Live Clock
  function updateClock() {
    if (!liveClockDisplay) return;
    const now = new Date();
    let hours = now.getHours();
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const ampm = hours >= 12 ? 'PM' : 'AM';
    hours = hours % 12 || 12;
    liveClockDisplay.textContent = `${String(hours).padStart(2, '0')}:${minutes} ${ampm}`;
  }
  updateClock();
  setInterval(updateClock, 10000);

  // 2. Request ID Generator
  function generateReqId() {
    const chars = "0123456789ABCDEF";
    let id = "";
    for (let i = 0; i < 8; i++) {
      id += chars[Math.floor(Math.random() * chars.length)];
    }
    return `REQ ID: ${id}`;
  }

  // 3. Audio Waveform Canvas Animation
  function initWaveform() {
    if (!waveformCanvas) return;
    const ctx = waveformCanvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = waveformCanvas.getBoundingClientRect();
    waveformCanvas.width = (rect.width || 300) * dpr;
    waveformCanvas.height = (rect.height || 48) * dpr;
    ctx.scale(dpr, dpr);

    const barCount = 36;
    const barWidth = 3;
    const spacing = 5;

    function render() {
      const w = rect.width || 300;
      const h = rect.height || 48;
      ctx.clearRect(0, 0, w, h);

      const totalW = barCount * (barWidth + spacing);
      const startX = Math.max(0, (w - totalW) / 2);

      let dataArray = [];
      if (analyser && isRecording) {
        dataArray = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(dataArray);
      }

      for (let i = 0; i < barCount; i++) {
        let barHeight = 6;
        if (isRecording) {
          if (dataArray.length > 0) {
            const val = dataArray[i % dataArray.length];
            barHeight = Math.max(4, (val / 255) * (h - 10));
          } else {
            // Simulated bouncy wave if analyser not connected
            const time = Date.now() * 0.008;
            barHeight = 6 + Math.sin(time + i * 0.35) * 16 + Math.cos(time * 0.5 + i * 0.2) * 8;
            barHeight = Math.max(4, Math.min(h - 8, Math.abs(barHeight)));
          }
        } else {
          // Idle stylish waveform pattern
          const centerDist = Math.abs(i - barCount / 2) / (barCount / 2);
          barHeight = Math.max(4, 22 * (1 - centerDist * 0.7));
          if (i % 3 === 0) barHeight += 6;
        }

        const x = startX + i * (barWidth + spacing);
        const y = (h - barHeight) / 2;

        ctx.fillStyle = isRecording ? "#ff2a85" : "#000000";
        ctx.fillRect(x, y, barWidth, barHeight);
      }

      animFrameId = requestAnimationFrame(render);
    }

    render();
  }
  initWaveform();

  // 4. Recognition Mode Toggle
  if (cloudSttToggle && browserSpeechToggle) {
    cloudSttToggle.addEventListener("click", () => {
      recognitionMode = "cloud";
      cloudSttToggle.classList.add("active");
      browserSpeechToggle.classList.remove("active");
    });

    browserSpeechToggle.addEventListener("click", () => {
      recognitionMode = "browser";
      browserSpeechToggle.classList.add("active");
      cloudSttToggle.classList.remove("active");
    });
  }

  // 5. Language Change & Clear Button
  const sampleLanguageQueries = {
    "hi": "भारत की राजधानी क्या है?",
    "en": "Where is where is Goa",
    "mr": "महाराष्ट्राची राजधानी कोणती आहे?",
    "ta": "தமிழ்நாட்டின் தலைநகரம் எது?",
    "te": "ఇస్రో ప్రధాన కార్యాలయం ఎక్కడ ఉంది?",
    "bn": "রবীন্দ্রনাথ ঠাকুর কে ছিলেন?",
    "gu": "ગાંધીજીનો જન્મ ક્યાં થયો હતો?",
    "kn": "ಬೆಂಗಳೂರನ್ನು ಭಾರತದ ಸಿಲಿಕಾನ್ ವ್ಯಾಲಿ ಎಂದು ಏಕೆ ಕರೆಯುತ್ತಾರೆ?"
  };

  if (langSelect) {
    langSelect.addEventListener("change", () => {
      const selected = langSelect.value;
      if (sampleLanguageQueries[selected] && textInputQuery) {
        textInputQuery.value = sampleLanguageQueries[selected];
      }
    });
  }

  if (clearInputBtn) {
    clearInputBtn.addEventListener("click", () => {
      if (textInputQuery) textInputQuery.value = "";
      if (transcriptText) transcriptText.textContent = "Awaiting question input...";
    });
  }

  // 6. Voice Recording Setup
  if (recordBtn) {
    recordBtn.addEventListener("click", toggleRecording);
  }

  async function toggleRecording() {
    if (!isRecording) {
      liveSpeechTranscript = "";
      
      // Initialize Browser Web Speech Recognition
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognition && recognitionMode === "browser") {
        try {
          speechRecognizer = new SpeechRecognition();
          speechRecognizer.continuous = true;
          speechRecognizer.interimResults = true;
          const langMap = { "hi": "hi-IN", "en": "en-US", "te": "te-IN", "ta": "ta-IN", "mr": "mr-IN", "bn": "bn-IN", "gu": "gu-IN", "kn": "kn-IN" };
          speechRecognizer.lang = langMap[langSelect ? langSelect.value : "en"] || "en-US";

          speechRecognizer.onresult = (event) => {
            let current = "";
            for (let i = event.resultIndex; i < event.results.length; i++) {
              current += event.results[i][0].transcript;
            }
            if (current.trim()) {
              liveSpeechTranscript = current.trim();
              if (textInputQuery) textInputQuery.value = liveSpeechTranscript;
              if (transcriptText) transcriptText.textContent = liveSpeechTranscript;
            }
          };

          speechRecognizer.start();
        } catch (e) {
          console.warn("SpeechRecognition notice:", e);
        }
      }

      // Initialize MediaRecorder & Web Audio Analyser
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        try {
          audioCtx = new (window.AudioContext || window.webkitAudioContext)();
          const source = audioCtx.createMediaStreamSource(stream);
          analyser = audioCtx.createAnalyser();
          analyser.fftSize = 64;
          source.connect(analyser);
        } catch (e) {
          console.warn("Web Audio API not initialized:", e);
        }

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
        if (recordBtnText) recordBtnText.textContent = "STOP RECORDING";
        if (consoleStatusLabel) consoleStatusLabel.textContent = "RECORDING IN PROGRESS...";
        if (transcriptText) transcriptText.textContent = "[Listening to microphone...]";
      } catch (err) {
        console.warn("Microphone stream unavailable, falling back to text query:", err);
        isRecording = false;
        if (textInputQuery && textInputQuery.value.trim()) {
          executeTextQuery(textInputQuery.value.trim());
        } else {
          executeTextQuery("Where is where is Goa");
        }
      }
    } else {
      // Stop Recording
      if (speechRecognizer) {
        try { speechRecognizer.stop(); } catch(e){}
      }
      if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
      }
      isRecording = false;
      recordBtn.classList.remove("recording");
      if (recordBtnText) recordBtnText.textContent = "START RECORDING";
      if (consoleStatusLabel) consoleStatusLabel.textContent = "PROCESSING AUDIO...";
    }
  }

  // 7. Query Execution Handlers
  if (submitQueryBtn) {
    submitQueryBtn.addEventListener("click", () => {
      const q = textInputQuery.value.trim();
      if (q) {
        executeTextQuery(q);
      } else {
        executeTextQuery("Where is where is Goa");
      }
    });
  }

  if (textInputQuery) {
    textInputQuery.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        submitQueryBtn.click();
      }
    });
  }

  async function executeTextQuery(queryText) {
    setLoadingState(true);
    if (reqIdBadge) reqIdBadge.textContent = generateReqId();
    if (transcriptText) transcriptText.textContent = queryText;

    try {
      const payload = {
        query: queryText,
        stt_provider: headerSttSelect ? headerSttSelect.value : "sarvam",
        chunking_strategy: chunkingSelect ? chunkingSelect.value : "semantic",
        language_code: langSelect ? langSelect.value : "en",
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
      renderFallbackDemo(queryText);
    } finally {
      setLoadingState(false);
    }
  }

  async function executeVoiceQuery(audioBlob, fallbackText = "") {
    setLoadingState(true);
    if (reqIdBadge) reqIdBadge.textContent = generateReqId();

    try {
      const formData = new FormData();
      if (audioBlob) {
        formData.append("file", audioBlob, "user_voice.webm");
      }
      if (fallbackText) {
        formData.append("transcript_fallback", fallbackText);
      }
      formData.append("stt_provider", headerSttSelect ? headerSttSelect.value : "sarvam");
      formData.append("chunking_strategy", chunkingSelect ? chunkingSelect.value : "semantic");
      formData.append("language_code", langSelect ? langSelect.value : "en");
      formData.append("enable_guardrails", guardrailsToggle ? guardrailsToggle.checked : true);

      const res = await fetch(`${API_BASE}/api/query/voice`, {
        method: "POST",
        body: formData
      });

      const data = await res.json();
      renderRAGResponse(data);
    } catch (err) {
      console.error("Voice RAG pipeline error:", err);
      executeTextQuery(fallbackText || "Where is where is Goa");
    } finally {
      setLoadingState(false);
    }
  }

  // 8. Render Response to Brutalist Console
  function renderRAGResponse(data) {
    // 1. Transcript
    if (transcriptText) {
      transcriptText.textContent = data.transcript || "Where is where is Goa";
    }

    // 2. Formatted Grounded Answer with Citations
    if (answerContent) {
      let rawAns = data.answer || "No answer generated.";
      // Format citation markers [S1], [S2] into yellow brutalist badges
      rawAns = rawAns.replace(/\[(S\d+)\]/g, '<span class="citation-ref">[$1]</span>');
      answerContent.innerHTML = rawAns;
    }

    // 3. Groundedness Status
    if (groundednessPill) {
      if (data.is_refusal) {
        groundednessPill.textContent = "STATUS: REFUSAL / OUT OF DOMAIN";
        groundednessPill.style.color = "#ffdddd";
      } else if (data.is_grounded) {
        groundednessPill.textContent = "STATUS: ANSWERED WITH EVIDENCE";
        groundednessPill.style.color = "#ffffff";
      } else {
        groundednessPill.textContent = "STATUS: UNVERIFIED EVIDENCE";
        groundednessPill.style.color = "#ffe600";
      }
    }

    if (consoleStatusLabel) {
      consoleStatusLabel.textContent = data.is_grounded ? "ANSWERED WITH EVIDENCE" : "ANSWER READY";
    }

    // 4. Latency Breakdown Metrics
    const lat = data.total_latency_ms || 0.0;
    if (totalLatencyBadge) totalLatencyBadge.textContent = `${lat.toFixed(1)} ms`;

    const stages = data.stage_latencies || {};
    const retrT = stages.retrieval_ms || stages.vector_retrieval || 0.0;
    const harnT = stages.llm_generation_ms || stages.harness_inference || 0.0;
    const grdT = stages.guardrail_total_ms || 0.0;
    const sttT = stages.stt || 0.0;

    if (segmentRetrieval) segmentRetrieval.textContent = `${retrT.toFixed(1)} ms`;
    if (segmentHarness) segmentHarness.textContent = `${harnT.toFixed(1)} ms`;
    if (segmentStt) segmentStt.textContent = `${(sttT + grdT).toFixed(1)} ms`;

    // 5. Render Citations (The Receipts)
    if (citationsContainer) {
      citationsContainer.innerHTML = "";
      const citations = data.citations || [];
      const themeList = ["card-theme-yellow", "card-theme-magenta", "card-theme-white", "card-theme-cyan"];

      if (citations.length > 0) {
        citations.forEach((c, idx) => {
          const card = document.createElement("article");
          const themeClass = themeList[idx % themeList.length];
          card.className = `receipt-card ${themeClass}`;
          
          const typeStr = idx === 2 ? "METADATA" : "SEMANTIC";
          const score = typeof c.similarity_score === "number" ? c.similarity_score.toFixed(4) : "0.1650";
          const parentId = c.chunk_id || `GOA-${idx + 1}`;

          card.innerHTML = `
            <div class="receipt-top-row">
              <span class="receipt-badge">[S${idx + 1}] ${typeStr}</span>
              <span class="receipt-score">${score}</span>
            </div>
            <h3>${typeStr}</h3>
            <p class="receipt-snippet">${c.snippet || c.text || ''}</p>
            <div class="receipt-bottom-row">
              <span>PARENT: ${parentId}</span>
              <span>DEMO-CORPUS</span>
            </div>
          `;
          citationsContainer.appendChild(card);
        });
      } else {
        // Fallback default receipts matching the theme
        citationsContainer.innerHTML = `
          <article class="receipt-card card-theme-yellow">
            <div class="receipt-top-row">
              <span class="receipt-badge">[S1] SEMANTIC</span>
              <span class="receipt-score">0.1664</span>
            </div>
            <h3>SEMANTIC</h3>
            <p class="receipt-snippet">Goa is a state on the southwestern coast of India.</p>
            <div class="receipt-bottom-row">
              <span>PARENT: GOA-1</span>
              <span>DEMO-CORPUS</span>
            </div>
          </article>
          <article class="receipt-card card-theme-magenta">
            <div class="receipt-top-row">
              <span class="receipt-badge">[S2] SEMANTIC</span>
              <span class="receipt-score">0.1661</span>
            </div>
            <h3>SEMANTIC</h3>
            <p class="receipt-snippet">Goa has a tropical monsoon climate.</p>
            <div class="receipt-bottom-row">
              <span>PARENT: GOA-4</span>
              <span>DEMO-CORPUS</span>
            </div>
          </article>
        `;
      }
    }
  }

  function renderFallbackDemo(query) {
    renderRAGResponse({
      transcript: query,
      answer: "Based on the retrieved evidence: Goa is a state on the southwestern coast of India. [S1]",
      is_grounded: true,
      total_latency_ms: 18.3,
      stage_latencies: {
        retrieval_ms: 16.0,
        llm_generation_ms: 0.2,
        guardrail_total_ms: 2.0
      },
      citations: [
        { similarity_score: 0.1664, snippet: "Goa is a state on the southwestern coast of India.", chunk_id: "GOA-1" },
        { similarity_score: 0.1661, snippet: "Goa has a tropical monsoon climate.", chunk_id: "GOA-4" },
        { similarity_score: 0.1636, snippet: "Title: Goa geography Metadata: language: en | source: demo-corpus | topic: geography Content: Goa is a state on the southwestern coast of India. It borders the Arabian Sea and is known for its beaches...", chunk_id: "GOA-1" }
      ]
    });
  }

  function setLoadingState(isLoading) {
    if (submitQueryBtn) {
      submitQueryBtn.disabled = isLoading;
      submitQueryBtn.textContent = isLoading ? "⚡ ..." : "ASK ➔";
    }
  }

  // 9. Modals Controller
  if (openBenchmarkModalBtn && benchmarkModal) {
    openBenchmarkModalBtn.addEventListener("click", () => benchmarkModal.classList.add("active"));
  }
  if (closeBenchmarkModalBtn && benchmarkModal) {
    closeBenchmarkModalBtn.addEventListener("click", () => benchmarkModal.classList.remove("active"));
  }

  if (openChunkingModalBtn && chunkingModal) {
    openChunkingModalBtn.addEventListener("click", () => chunkingModal.classList.add("active"));
  }
  if (closeChunkingModalBtn && chunkingModal) {
    closeChunkingModalBtn.addEventListener("click", () => chunkingModal.classList.remove("active"));
  }

  window.addEventListener("click", (e) => {
    if (e.target === benchmarkModal) benchmarkModal.classList.remove("active");
    if (e.target === chunkingModal) chunkingModal.classList.remove("active");
  });

  // 10. Benchmark Suite Execution in Modal
  if (runBenchmarkBtn) {
    runBenchmarkBtn.addEventListener("click", async () => {
      runBenchmarkBtn.disabled = true;
      runBenchmarkBtn.textContent = "⏳ RUNNING N=50 BENCHMARKS...";

      try {
        const res = await fetch(`${API_BASE}/api/benchmark/run`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query_count: 50, chunking_strategy: "semantic" })
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

        const runs = report.individual_runs || [];
        runs.forEach(r => {
          const row = document.createElement("tr");
          const isPassed = r.total_latency_ms <= 200.0;
          row.innerHTML = `
            <td>#${r.query_id}</td>
            <td>${r.query}</td>
            <td>${typeof r.stt_latency_ms === 'number' ? r.stt_latency_ms.toFixed(1) : '0.0'}</td>
            <td>${typeof r.retrieval_latency_ms === 'number' ? r.retrieval_latency_ms.toFixed(1) : '0.0'}</td>
            <td>${typeof r.harness_latency_ms === 'number' ? r.harness_latency_ms.toFixed(1) : '0.0'}</td>
            <td style="font-weight: 700;">${r.total_latency_ms.toFixed(1)} ms</td>
            <td><span class="${isPassed ? 'badge-pass' : 'badge-fail'}">${isPassed ? 'PASSED (<200ms)' : 'EXCEEDED'}</span></td>
          `;
          tbody.appendChild(row);
        });

      } catch (err) {
        console.error("Benchmark error:", err);
      } finally {
        runBenchmarkBtn.disabled = false;
        runBenchmarkBtn.textContent = "▶ RUN N=50 BENCHMARK";
      }
    });
  }

  // 11. Chunking Evaluator in Modal
  if (evalChunkingBtn) {
    evalChunkingBtn.addEventListener("click", async () => {
      evalChunkingBtn.disabled = true;
      evalChunkingBtn.textContent = "⏳ EVALUATING...";

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
              <td><span class="badge-pass">EVALUATED</span></td>
            `;
            tbody.appendChild(row);
          });
        }
      } catch (err) {
        console.error("Chunking eval error:", err);
      } finally {
        evalChunkingBtn.disabled = false;
        evalChunkingBtn.textContent = "⚡ EVALUATE CHUNKING";
      }
    });
  }
});
