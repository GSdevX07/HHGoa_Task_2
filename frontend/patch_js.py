import sys
import re

file_path = r'D:\Vedhanth\studies\Coding\Hackathon\HH Goa\HHGoa_Task_2\frontend\app.js'
with open(file_path, 'r', encoding='utf-8') as f:
    js = f.read()

# Replace the citations logic to match the new brutalist classes
citations_replacement = """
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
"""

# Very careful regex
js = re.sub(r'// Citations.*?\}\n', citations_replacement, js, flags=re.DOTALL)

# Replace latency display
latency_replacement = """
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
"""

js = re.sub(r'// Total Latency & SLA Badge.*?segmentHarness\.textContent = `Harness: \$\{harnT\.toFixed\(1\)\}ms`;', latency_replacement, js, flags=re.DOTALL)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(js)
print("JS Patched!")
