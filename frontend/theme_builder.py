import os

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>VAANI RAG STUDIO GOA</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Anton&family=Space+Mono:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <div class="brutalist-container">
    
    <!-- LEFT SIDEBAR -->
    <div class="col-left">
      <div class="logo-area">
        <span class="logo-icon">✶</span>
        <div>
          <h1 class="logo-text">VAANI</h1>
          <p class="logo-sub">RAG STUDIO • GOA</p>
        </div>
      </div>

      <div class="brutal-box yellow-box tilt-box" style="position: relative;">
        <span class="task-badge">TASK #02</span>
        <h2>VOICE-ENABLED<br>RAG MODEL</h2>
        <p class="desc-text">Speak a question, get a grounded answer. Transcription, engineered chunking, vector retrieval, and generation — wired together end to end.</p>
      </div>

      <div class="status-area">
        <div class="status-row"><span>STATUS</span> <span class="accent-green">READY</span></div>
        <div class="status-desc">ELEVENLABS STT • ANTHROPIC PIPELINE/CLAUDE / CLAUDE-3-5-SONNET-20241022</div>
        <div class="status-row"><span>DATE</span> <span>28-31 OCT 2026</span></div>
        <div class="status-row"><span>TIME</span> <span>01:48 PM</span></div>
      </div>
      
      <div class="badges">
        <span class="badge white-badge">SAFE BY DESIGN</span>
        <span class="badge magenta-badge">NO HALLUCINATIONS +</span>
      </div>
    </div>

    <!-- CENTER CONSOLE -->
    <div class="col-center">
      <div class="yellow-marquee brutal-border">
        <span>+ TASK 02 + VOICE RAG</span>
        <span>BUILD + LEARN + IMPACT</span>
        <span>+ ASK THE INDEX</span>
        <span>CHAI POWERED</span>
      </div>

      <div class="console-box brutal-border bg-white">
        <div class="console-header yellow-box brutal-border-bottom">
          <span class="dot">●</span> ANSWERED WITH EVIDENCE
          <h2 class="center-title">VOICE CONSOLE</h2>
          <span class="req-id">REQ ID: C6FC997A</span>
        </div>

        <div class="console-body">
          <div class="recording-bar brutal-border">
            <button id="recordBtn" class="record-btn bg-green brutal-border">
              <span class="red-dot">●</span> <span id="recordBtnText">START RECORDING</span>
            </button>
            <div class="waveform" id="waveformVisualizer">|||||||||||||||||||</div>
          </div>

          <div class="mode-row">
            <span class="label">RECOGNITION MODE</span>
            <div class="mode-toggles">
              <span class="toggle active bg-green">CLOUD STT</span>
              <span class="toggle">BROWSER SPEECH</span>
            </div>
          </div>

          <div class="dropdown-row">
            <select id="langSelect" class="brutal-input">
              <option value="en">AUTO-DETECT LANGUAGE</option>
              <option value="hi">HINDI</option>
            </select>
            <button class="clear-btn brutal-input" onclick="document.getElementById('textInputQuery').value='';">CLEAR ↺</button>
            <select id="chunkingSelect" style="display:none;"><option value="semantic">Semantic</option></select>
            <select id="headerSttSelect" style="display:none;"><option value="sarvam">Sarvam</option></select>
            <input type="checkbox" id="guardrailsToggle" checked style="display:none;">
          </div>

          <div class="preview-box">
            <span class="label">TRANSCRIPT PREVIEW</span>
            <div class="transcript-text brutal-border" id="transcriptText">
              Where is where is Goa
            </div>
          </div>

          <div class="query-fallback-row">
            <input type="text" id="textInputQuery" class="brutal-input flex-grow" placeholder="TYPE YOUR QUERY FALLBACK HERE...">
            <button id="submitQueryBtn" class="ask-btn brutal-border bg-black text-white">ASK +</button>
            <button id="sampleQueryBtn" style="display:none;">Sample</button>
          </div>
        </div>
      </div>

      <div class="answer-box brutal-border bg-white mt-20">
        <div class="answer-header magenta-box brutal-border-bottom">
          <h2>LIVE GROUNDED ANSWER</h2>
          <span class="status-right" id="groundednessPill">STATUS: WAITING</span>
        </div>
        <div class="answer-body">
          <div class="answer-text brutal-border" id="answerContent">
            Waiting for voice input...
          </div>
          
          <div class="latency-boxes">
            <div class="lat-box brutal-border"><span class="val" id="totalLatencyBadge">0.0 ms</span><span class="lbl">TOTAL</span></div>
            <div class="lat-box brutal-border"><span class="val" id="segmentRetrieval">0.0 ms</span><span class="lbl">RETRIEVAL</span></div>
            <div class="lat-box brutal-border"><span class="val" id="segmentHarness">0.0 ms</span><span class="lbl">GENERATION</span></div>
            <div class="lat-box brutal-border"><span class="val" id="segmentStt">0.0 ms</span><span class="lbl">STT / GUARDRAILS</span></div>
          </div>
        </div>
      </div>
    </div>

    <!-- RIGHT SIDEBAR -->
    <div class="col-right">
      <div class="right-header">
        <span>02 / CITED EVIDENCE</span>
        <span>THE RECEIPTS</span>
      </div>
      
      <div id="citationsContainer" class="citations-list">
        <!-- Citations will be injected here by app.js -->
        <div class="citation-card yellow-box brutal-border">
          <div class="cit-top"><span>[S1] SEMANTIC</span><span class="cit-score">0.1664</span></div>
          <h3>SEMANTIC</h3>
          <p>Goa is a state on the southwestern coast of India.</p>
          <div class="cit-bottom"><span>PARENT GOA-1</span><span>DEMO-CORPUS</span></div>
        </div>
      </div>
      <div id="executionTraceContainer" style="display:none;"></div>
      
      <!-- Hidden buttons to prevent app.js from breaking -->
      <button id="runBenchmarkBtn" style="display:none;"></button>
      <button id="evalChunkingBtn" style="display:none;"></button>
      <button id="uploadAudioBtn" style="display:none;"></button>
      <input type="file" id="audioFileInput" style="display:none;">
    </div>
  </div>

  <script src="app.js"></script>
</body>
</html>
"""

CSS_CONTENT = """
:root {
  --bg-green: #0E793C;
  --yellow: #FFE800;
  --magenta: #FF0066;
  --white: #FFFFFF;
  --black: #000000;
  --border-thick: 3px solid #000;
  --shadow-thick: 4px 4px 0 #000;
  --font-heavy: 'Anton', sans-serif;
  --font-mono: 'Space Mono', monospace;
}

* { box-sizing: border-box; }

body {
  margin: 0; padding: 20px;
  background-color: var(--bg-green);
  font-family: var(--font-mono);
  color: var(--black);
  display: flex;
  justify-content: center;
  min-height: 100vh;
}

.brutalist-container {
  display: grid;
  grid-template-columns: 280px 1fr 320px;
  gap: 30px;
  max-width: 1400px;
  width: 100%;
}

/* Utilities */
.brutal-border { border: var(--border-thick); }
.brutal-border-bottom { border-bottom: var(--border-thick); }
.yellow-box { background: var(--yellow); }
.magenta-box { background: var(--magenta); color: var(--white); }
.bg-white { background: var(--white); }
.bg-green { background: var(--bg-green); color: var(--white); }
.bg-black { background: var(--black); color: var(--white); }
.text-white { color: var(--white); }
.mt-20 { margin-top: 20px; }

/* Left Col */
.col-left { display: flex; flex-direction: column; gap: 40px; }
.logo-area { display: flex; align-items: center; gap: 10px; color: var(--yellow); }
.logo-icon { font-size: 32px; }
.logo-text { font-family: var(--font-heavy); font-size: 42px; margin: 0; letter-spacing: 2px; }
.logo-sub { margin: 0; font-size: 12px; letter-spacing: 1px; }

.tilt-box { transform: rotate(-2deg); padding: 20px; box-shadow: var(--shadow-thick); border: var(--border-thick); }
.task-badge { position: absolute; top: -12px; right: 10px; background: var(--magenta); color: var(--white); padding: 2px 8px; font-weight: bold; border: 2px solid #000; font-size: 12px; }
.tilt-box h2 { font-family: var(--font-heavy); font-size: 32px; line-height: 1.1; margin: 0 0 15px 0; }
.desc-text { font-size: 13px; line-height: 1.4; margin: 0; }

.status-area { color: var(--yellow); font-size: 12px; display: flex; flex-direction: column; gap: 8px; }
.status-row { display: flex; justify-content: space-between; border-bottom: 1px dashed rgba(255,255,255,0.3); padding-bottom: 4px; }
.accent-green { color: #A8FF53; }
.status-desc { font-size: 10px; opacity: 0.8; text-align: right; margin-bottom: 10px; }

.badges { display: flex; gap: 10px; }
.badge { padding: 4px 10px; font-size: 11px; font-weight: bold; border: 2px solid #000; border-radius: 20px; }
.white-badge { background: var(--white); color: var(--black); }
.magenta-badge { background: var(--magenta); color: var(--white); }

/* Center Col */
.col-center { display: flex; flex-direction: column; }
.yellow-marquee { background: var(--yellow); padding: 8px 15px; font-family: var(--font-heavy); display: flex; justify-content: space-between; font-size: 18px; margin-bottom: 20px; box-shadow: var(--shadow-thick); }

.console-box, .answer-box { box-shadow: var(--shadow-thick); }
.console-header, .answer-header { display: flex; justify-content: space-between; align-items: center; padding: 10px 20px; }
.console-header h2, .answer-header h2 { font-family: var(--font-heavy); font-size: 24px; margin: 0; }
.dot { color: var(--magenta); margin-right: 8px; }
.req-id, .status-right { font-size: 12px; font-weight: bold; }

.console-body, .answer-body { padding: 20px; }
.recording-bar { display: flex; align-items: stretch; margin-bottom: 20px; background: #FFF; box-shadow: 2px 2px 0 #000; }
.record-btn { border: none; border-right: var(--border-thick); font-family: var(--font-heavy); font-size: 28px; padding: 15px 30px; cursor: pointer; display: flex; align-items: center; gap: 15px; }
.red-dot { color: var(--magenta); font-size: 24px; }
.waveform { flex-grow: 1; display: flex; align-items: center; justify-content: center; font-size: 24px; letter-spacing: 2px; }

.mode-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; font-size: 12px; font-weight: bold; }
.mode-toggles { display: flex; border: 2px solid #000; }
.toggle { padding: 4px 12px; border-right: 2px solid #000; cursor: pointer; }
.toggle:last-child { border-right: none; }

.dropdown-row { display: flex; gap: 10px; margin-bottom: 15px; }
.brutal-input { border: 2px solid #000; padding: 8px 12px; font-family: var(--font-mono); font-size: 12px; background: transparent; font-weight: bold; }
.clear-btn { cursor: pointer; background: #f0f0f0; }

.label { font-size: 10px; font-weight: bold; color: #666; margin-bottom: 5px; display: block; }
.transcript-text { padding: 15px; font-size: 18px; font-weight: bold; min-height: 60px; margin-bottom: 20px; box-shadow: 2px 2px 0 #000; }

.query-fallback-row { display: flex; gap: 10px; }
.flex-grow { flex-grow: 1; }
.ask-btn { font-family: var(--font-heavy); font-size: 20px; padding: 0 20px; cursor: pointer; box-shadow: 2px 2px 0 #000; }

.answer-text { padding: 20px; font-size: 22px; font-weight: bold; min-height: 120px; margin-bottom: 20px; box-shadow: 2px 2px 0 #000; }

.latency-boxes { display: flex; gap: 15px; }
.lat-box { flex: 1; padding: 10px; display: flex; flex-direction: column; box-shadow: 2px 2px 0 #000; }
.lat-box .val { font-family: var(--font-heavy); font-size: 20px; }
.lat-box .lbl { font-size: 10px; font-weight: bold; color: #666; margin-top: 5px; }

/* Right Col */
.right-header { display: flex; justify-content: space-between; color: var(--white); font-size: 12px; font-weight: bold; margin-bottom: 15px; border-bottom: 2px solid var(--white); padding-bottom: 5px; }

.citations-list { display: flex; flex-direction: column; gap: 20px; }
.citation-card { padding: 15px; box-shadow: var(--shadow-thick); }
.citation-card:nth-child(1) { background: var(--yellow); color: var(--black); }
.citation-card:nth-child(2) { background: var(--magenta); color: var(--white); }
.citation-card:nth-child(3) { background: var(--white); color: var(--black); }

.cit-top { display: flex; justify-content: space-between; font-size: 10px; font-weight: bold; background: var(--black); color: var(--white); padding: 2px 6px; margin-bottom: 10px; display: inline-flex; gap: 10px; }
.citation-card h3 { font-size: 24px; margin: 0 0 10px 0; }
.citation-card p { font-size: 12px; line-height: 1.4; margin: 0 0 15px 0; }
.cit-bottom { display: flex; justify-content: space-between; font-size: 10px; font-weight: bold; border-top: 2px dashed rgba(0,0,0,0.3); padding-top: 10px; }
.citation-card:nth-child(2) .cit-bottom { border-top-color: rgba(255,255,255,0.4); }

"""

JS_PATCH = """
import sys
import re

file_path = r'D:\\Vedhanth\\studies\\Coding\\Hackathon\\HH Goa\\HHGoa_Task_2\\frontend\\app.js'
with open(file_path, 'r', encoding='utf-8') as f:
    js = f.read()

# Replace the citations logic to match the new brutalist classes
citations_replacement = '''
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
'''

js = re.sub(r'// Citations.*?\} else \{.*?\n\s*\}', citations_replacement, js, flags=re.DOTALL)

# Replace latency display
latency_replacement = '''
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
'''

js = re.sub(r'// Total Latency & SLA Badge.*?segmentHarness.textContent = `Harness: \$\{harnT\.toFixed\(1\)\}ms`;', latency_replacement, js, flags=re.DOTALL)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(js)
print("JS Patched!")
"""

with open(r'D:\Vedhanth\studies\Coding\Hackathon\HH Goa\HHGoa_Task_2\frontend\index.html', 'w', encoding='utf-8') as f:
    f.write(HTML_CONTENT)

with open(r'D:\Vedhanth\studies\Coding\Hackathon\HH Goa\HHGoa_Task_2\frontend\styles.css', 'w', encoding='utf-8') as f:
    f.write(CSS_CONTENT)

with open(r'D:\Vedhanth\studies\Coding\Hackathon\HH Goa\HHGoa_Task_2\frontend\patch_js.py', 'w', encoding='utf-8') as f:
    f.write(JS_PATCH)
