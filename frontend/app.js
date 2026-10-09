// Research QA System — frontend logic
// All API calls are same-origin (FastAPI serves this file), so no CORS setup needed.

let allPapers = [];
let selectedPaperIds = new Set();

// ─────────────────────────────────────────────────────────────
// Papers list
// ─────────────────────────────────────────────────────────────

async function loadPapers() {
  const listEl = document.getElementById("papers-list");
  listEl.innerHTML = '<p class="muted">Loading papers…</p>';

  try {
    const resp = await fetch("/papers");
    const data = await resp.json();
    allPapers = data.papers || [];
    renderPapersList();
  } catch (err) {
    listEl.innerHTML = `<p class="muted">Failed to load papers: ${err.message}</p>`;
  }
}

function renderPapersList() {
  const listEl = document.getElementById("papers-list");

  if (allPapers.length === 0) {
    listEl.innerHTML = '<p class="muted">No papers yet — upload one above.</p>';
    return;
  }

  listEl.innerHTML = "";
  for (const paper of allPapers) {
    const card = document.createElement("div");
    card.className = "paper-card" + (selectedPaperIds.has(paper.paper_id) ? " selected" : "");
    card.dataset.paperId = paper.paper_id;

    const subjectBadge = paper.subject_name
      ? `<span class="subject-badge">${escapeHtml(paper.subject_name)}</span>`
      : `<span class="subject-badge none">no subject detected</span>`;

    card.innerHTML = `
      <div class="paper-title">${escapeHtml(paper.title)}</div>
      <div class="paper-meta">id ${paper.paper_id} · ${paper.num_pages} pages</div>
      ${subjectBadge}
    `;

    card.addEventListener("click", () => togglePaperSelection(paper.paper_id));
    listEl.appendChild(card);
  }
}

function togglePaperSelection(paperId) {
  if (selectedPaperIds.has(paperId)) {
    selectedPaperIds.delete(paperId);
  } else {
    selectedPaperIds.add(paperId);
  }
  renderPapersList();
  renderSelectedPapersDisplay();
  if (document.getElementById("filter-history-checkbox").checked) {
    loadHistory();
  }
}

function renderSelectedPapersDisplay() {
  const el = document.getElementById("selected-papers-display");
  if (selectedPaperIds.size === 0) {
    el.textContent = "none — select from the left";
    el.className = "muted";
    return;
  }
  const names = [...selectedPaperIds].map((id) => {
    const p = allPapers.find((p) => p.paper_id === id);
    return p ? (p.subject_name || p.title) : `#${id}`;
  });
  el.textContent = names.join(", ");
  el.className = "";
}

// ─────────────────────────────────────────────────────────────
// History list
// ─────────────────────────────────────────────────────────────
let allHistory = [];

async function loadHistory() {
  const listEl = document.getElementById("history-list");
  listEl.innerHTML = '<p class="muted">Loading history…</p>';
  
  const filterSelected = document.getElementById("filter-history-checkbox").checked;
  let url = "/history";
  
  if (filterSelected && selectedPaperIds.size > 0) {
    // Only fetch for the first selected paper for simplicity, or we could pass multiple
    // The API currently accepts a single paper_id for filtering
    const firstPaperId = [...selectedPaperIds][0];
    url += `?paper_id=${firstPaperId}`;
  } else if (filterSelected) {
    listEl.innerHTML = '<p class="muted">No papers selected to filter by.</p>';
    return;
  }

  try {
    const resp = await fetch(url);
    const data = await resp.json();
    allHistory = data.history || [];
    renderHistoryList();
  } catch (err) {
    listEl.innerHTML = `<p class="muted">Failed to load history: ${err.message}</p>`;
  }
}

function renderHistoryList() {
  const listEl = document.getElementById("history-list");

  if (allHistory.length === 0) {
    listEl.innerHTML = '<p class="muted">No history found.</p>';
    return;
  }

  listEl.innerHTML = "";
  for (const item of allHistory) {
    const card = document.createElement("div");
    card.className = "history-card";
    
    const d = new Date(item.created_at);
    const timeStr = d.toLocaleDateString() + " " + d.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    
    // Join paper titles, defaulting to "Global Search" if none
    const papersStr = item.paper_titles && item.paper_titles.length > 0 
      ? item.paper_titles.join(", ") 
      : "Global Search";
      
    const typeBadge = item.is_baseline 
      ? `<span class="subject-badge none" style="background:#fef08a;color:#854d0e;">Baseline</span>`
      : `<span class="subject-badge" style="background:#dcfce7;color:#166534;">Verified</span>`;

    card.innerHTML = `
      <div class="history-question">${escapeHtml(item.question)}</div>
      <div class="history-meta">${timeStr} · ${escapeHtml(papersStr)}</div>
      ${typeBadge}
    `;

    card.addEventListener("click", () => loadHistoryDetail(item.id, card));
    listEl.appendChild(card);
  }
}

async function loadHistoryDetail(id, cardEl) {
  const resultsArea = document.getElementById("results-area");

  // Highlight selected card
  document.querySelectorAll('.history-card').forEach(c => c.classList.remove('selected'));
  if (cardEl) cardEl.classList.add('selected');
  
  setLoading(true, "Loading saved answer…");
  try {
    const resp = await fetch(`/history/${id}`);
    const data = await resp.json();
    
    if (!resp.ok) {
      throw new Error(data.detail || "Failed to load history item");
    }
    
    resultsArea.innerHTML = "";
    
    // Check if it was baseline by checking if citation_health exists (baseline doesn't have it)
    if (data.citation_health) {
      renderFullArchitectureResult(data);
    } else {
      renderBaselineResult(data);
    }
  } catch (err) {
    resultsArea.innerHTML = `<div class="error-box">Error: ${escapeHtml(err.message)}</div>`;
  } finally {
    setLoading(false);
  }
}

// ─────────────────────────────────────────────────────────────
// Upload
// ─────────────────────────────────────────────────────────────

async function uploadPdf() {
  const fileInput = document.getElementById("pdf-file-input");
  const statusEl = document.getElementById("upload-status");
  const uploadBtn = document.getElementById("upload-btn");

  if (!fileInput.files.length) {
    statusEl.textContent = "Choose a PDF file first.";
    statusEl.className = "status-text error";
    return;
  }

  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append("file", file);

  uploadBtn.disabled = true;
  statusEl.textContent = "Uploading and processing (this can take 15-30s)…";
  statusEl.className = "status-text info";

  try {
    const resp = await fetch("/ingest/pdf", { method: "POST", body: formData });
    const data = await resp.json();

    if (!resp.ok) {
      throw new Error(data.detail ? JSON.stringify(data.detail) : "Upload failed");
    }

    statusEl.textContent = `Uploaded: "${data.title}" (${data.num_pages} pages, id ${data.paper_id})`;
    statusEl.className = "status-text success";
    fileInput.value = "";
    await loadPapers();
  } catch (err) {
    statusEl.textContent = `Error: ${err.message}`;
    statusEl.className = "status-text error";
  } finally {
    uploadBtn.disabled = false;
  }
}

// ─────────────────────────────────────────────────────────────
// Ask
// ─────────────────────────────────────────────────────────────

function setLoading(isLoading, text) {
  const indicator = document.getElementById("loading-indicator");
  const loadingText = document.getElementById("loading-text");
  const askBtn = document.getElementById("ask-btn");

  indicator.classList.toggle("hidden", !isLoading);
  askBtn.disabled = isLoading;
  if (text) loadingText.textContent = text;
}

async function askQuestion() {
  const question = document.getElementById("question-input").value.trim();
  const resultsArea = document.getElementById("results-area");
  const compareBaseline = document.getElementById("compare-baseline-checkbox").checked;

  if (!question) {
    alert("Type a question first.");
    return;
  }
  if (selectedPaperIds.size === 0) {
    alert("Select at least one paper first.");
    return;
  }

  resultsArea.innerHTML = "";

  const paperIdParams = [...selectedPaperIds].map((id) => `paper_id=${id}`).join("&");

  setLoading(true, "Generating answer...");

  try {
    // Render the initial block
    resultsArea.innerHTML = `
      <div class="result-block full-arch" id="stream-block">
        <div class="result-header">
          <h3>Full Architecture</h3>
        </div>
        <div class="answer-text" id="stream-text"></div>
        <div id="stream-loading" class="muted" style="margin-top: 10px; font-size: 13px;">
           <span class="spinner"></span> Generating...
        </div>
      </div>
    `;
    const streamTextEl = document.getElementById("stream-text");
    const streamLoadingEl = document.getElementById("stream-loading");

    const fullResp = await fetch(`/answer/stream?q=${encodeURIComponent(question)}&${paperIdParams}`);
    
    if (!fullResp.ok) {
      const errData = await fullResp.json().catch(() => ({}));
      throw new Error(errData.detail || "Request failed");
    }

    const reader = fullResp.body.getReader();
    const decoder = new TextDecoder();
    let queryId = null;
    let eventType = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n');
      
      for (const line of lines) {
        if (line.startsWith('event: ')) {
          eventType = line.substring(7).trim();
        } else if (line.startsWith('data: ')) {
          const dataStr = line.substring(6).trim();
          if (!dataStr) continue;
          const data = JSON.parse(dataStr);
          
          if (eventType === 'metadata') {
            queryId = data.query_id;
          } else if (eventType === 'text') {
            streamTextEl.textContent += data.chunk;
          } else if (eventType === 'error') {
            throw new Error(data.error);
          }
        }
      }
    }

    // Now we have the queryId, start polling for verification
    streamLoadingEl.innerHTML = '<span class="spinner"></span> Verifying claims against evidence...';
    
    let finalData = null;
    while (true) {
      await new Promise(r => setTimeout(r, 1000)); // poll every 1s
      const pollResp = await fetch(`/answer/verify/${queryId}`);
      if (!pollResp.ok) {
         const errData = await pollResp.json().catch(() => ({}));
         throw new Error(errData.detail || "Verification failed");
      }
      const pollData = await pollResp.json();
      if (pollData.status === "done") {
         finalData = pollData.result;
         break;
      } else if (pollData.status === "error") {
         throw new Error(pollData.error || "Verification failed");
      }
    }

    // Replace the stream block with the full architecture block
    resultsArea.innerHTML = "";
    renderFullArchitectureResult(finalData);

    if (compareBaseline) {
      setLoading(true, "Running baseline for comparison…");
      const firstPaperId = [...selectedPaperIds][0];
      const baselineResp = await fetch(
        `/baseline/answer?q=${encodeURIComponent(question)}&paper_id=${firstPaperId}`
      );
      const baselineData = await baselineResp.json();
      if (baselineResp.ok) {
        renderBaselineResult(baselineData);
      }
    }
  } catch (err) {
    resultsArea.innerHTML += `<div class="error-box">Error: ${escapeHtml(err.message)}</div>`;
  } finally {
    setLoading(false);
    loadHistory();
  }
}

function renderFullArchitectureResult(data) {
  const resultsArea = document.getElementById("results-area");

  const block = document.createElement("div");
  block.className = "result-block full-arch";

  // ── Group claims by parent_claim_id ───────────────────────────────────
  // Each group is { parentId, subClaims[], aggregateStatus }.
  // Atomic claims (parent_claim_id == null) each form their own group of 1,
  // rendered exactly as before. Decomposed sub-claims are grouped together
  // under their parent and rendered with an expandable sub-list.
  const groups = _groupClaims(data.claims || []);

  const claimsHtml = groups.map(group => {
    if (group.isAtomic) {
      // ── Standard claim (unchanged from original rendering) ───────────
      const claim = group.subClaims[0];
      return _renderSingleClaimHtml(claim);
    } else {
      // ── Decomposed compound claim ────────────────────────────────────
      const statusClass = group.aggregateStatus === "VALID" ? "valid" : "flagged";
      const subClaimsHtml = group.subClaims.map(_renderSingleClaimHtml).join("");
      return `
        <div class="claim-item ${statusClass} decomposed-parent">
          <span class="claim-status-tag ${statusClass}">${group.aggregateStatus}</span>
          <span style="font-size:12px;color:#555;font-style:italic;">
            Compound claim decomposed into ${group.subClaims.length} sub-facts:
          </span>
          <details class="sub-claims-details">
            <summary class="sub-claims-summary">Show sub-facts ▸</summary>
            <div class="sub-claims-list">${subClaimsHtml}</div>
          </details>
        </div>`;
    }
  }).join("");

  const evidenceHtml = (data.evidence_used || [])
    .map((e) => `
      <details class="evidence-details">
        <summary class="evidence-tag clickable">[${e.label}] ${escapeHtml(e.paper_title)} p.${e.page}</summary>
        <div class="evidence-text">${escapeHtml(e.text)}</div>
      </details>
    `)
    .join("");

  const uncertaintiesHtml = (data.uncertainties || []).length
    ? `<div class="uncertainties"><strong>Uncertainties:</strong> ${data.uncertainties
        .map(escapeHtml)
        .join("; ")}</div>`
    : "";

  const warningsHtml = (data.resolution_warnings || []).length
    ? `<div class="uncertainties"><strong>Note:</strong> ${data.resolution_warnings
        .map(escapeHtml)
        .join("; ")}</div>`
    : "";

  block.innerHTML = `
    <div class="result-header">
      <h3>Full Architecture (verified)</h3>
      <span class="result-meta">subject_name used: ${data.subject_name_used || "none"}</span>
    </div>
    <div class="answer-text">${escapeHtml(data.answer)}</div>
    ${warningsHtml}
    ${uncertaintiesHtml}
    <div class="claims-section">
      <h4>Claims &amp; Verification</h4>
      ${claimsHtml || '<p class="muted">No claims extracted.</p>'}
    </div>
    <div class="citation-health-bar">
      Citation health: <strong>${data.citation_health.valid_claims}/${data.citation_health.total_claims}</strong> claims verified as directly supported by their cited evidence.
    </div>
    <div class="evidence-section">
      <h4>Evidence Used</h4>
      ${evidenceHtml}
    </div>
  `;

  resultsArea.appendChild(block);
}

// ── Helpers for grouped claim rendering ──────────────────────────────────────

function _groupClaims(claims) {
  // Pass 1: collect sub-claims by parent_claim_id (preserves insertion order)
  const parentOrder = [];  // ordered list of parent keys seen
  const groupMap = {};     // parentId -> subClaims[]

  for (const claim of claims) {
    const key = claim.parent_claim_id || claim.id;  // atomic: key = own id
    if (!groupMap[key]) {
      groupMap[key] = [];
      parentOrder.push(key);
    }
    groupMap[key].push(claim);
  }

  // Pass 2: compute aggregate status and isAtomic flag
  return parentOrder.map(key => {
    const subClaims = groupMap[key];
    const isAtomic = subClaims.length === 1 && subClaims[0].parent_claim_id == null;

    // ALL-VALID rule: parent is VALID only if every sub-fact is VALID.
    // NO_CITATIONS treated same as FLAGGED (non-VALID = parent not confirmed).
    const aggregateStatus = subClaims.every(c => c.verification_status === "VALID")
      ? "VALID"
      : "FLAGGED";

    return { isAtomic, subClaims, aggregateStatus };
  });
}

function _renderSingleClaimHtml(claim) {
  const statusClass =
    claim.verification_status === "VALID"
      ? "valid"
      : claim.verification_status === "FLAGGED"
      ? "flagged"
      : "no-citations";
  return `
    <div class="claim-item ${statusClass}">
      ${escapeHtml(claim.text)}
      <span class="claim-status-tag ${statusClass}">${claim.verification_status}</span>
      <div style="font-size:11px;color:#888;margin-top:4px;">
        citations: ${claim.citations.join(", ") || "none"}
        ${claim.joint_citation_check ? `<br/>joint check: ${claim.joint_citation_check.nli_label} (${(claim.joint_citation_check.confidence * 100).toFixed(0)}%)` : ""}
      </div>
    </div>`;
}

function renderBaselineResult(data) {
  const resultsArea = document.getElementById("results-area");

  const block = document.createElement("div");
  block.className = "result-block baseline";

  block.innerHTML = `
    <div class="result-header">
      <h3>Baseline (simple RAG — no verification)</h3>
      <span class="result-meta">no citation checking available</span>
    </div>
    <div class="answer-text">${escapeHtml(data.answer)}</div>
    <div class="citation-health-bar" style="color:#999;">
      This system cannot verify its own claims — treat any specific factual
      details with more caution than the verified answer above.
    </div>
  `;

  resultsArea.appendChild(block);
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

async function clearAllPapers() {
  if (allPapers.length === 0) {
    alert("No papers to clear.");
    return;
  }

  const confirmed = confirm(
    `This will permanently delete ALL ${allPapers.length} uploaded paper(s), ` +
    `including their sections, chunks, and PDF files. This cannot be undone.\n\n` +
    `Are you sure?`
  );
  if (!confirmed) return;

  const clearBtn = document.getElementById("clear-all-btn");
  clearBtn.disabled = true;
  clearBtn.textContent = "Clearing…";

  try {
    const resp = await fetch("/papers?confirm=true", { method: "DELETE" });
    const data = await resp.json();

    if (!resp.ok) {
      throw new Error(data.detail || "Failed to clear papers");
    }

    selectedPaperIds.clear();
    renderSelectedPapersDisplay();
    document.getElementById("results-area").innerHTML = "";
    await loadPapers();

    alert(`Cleared ${data.deleted_papers} paper(s).`);
  } catch (err) {
    alert(`Error clearing papers: ${err.message}`);
  } finally {
    clearBtn.disabled = false;
    clearBtn.textContent = "Clear all papers";
  }
}

// ─────────────────────────────────────────────────────────────
// Wire up
// ─────────────────────────────────────────────────────────────

document.getElementById("upload-btn").addEventListener("click", uploadPdf);
document.getElementById("refresh-papers-btn").addEventListener("click", loadPapers);
document.getElementById("clear-all-btn").addEventListener("click", clearAllPapers);
document.getElementById("ask-btn").addEventListener("click", askQuestion);
document.getElementById("filter-history-checkbox").addEventListener("change", loadHistory);

document.getElementById("clear-history-btn").addEventListener("click", async () => {
  if (!confirm("Clear all query history? This cannot be undone.")) return;

  const btn = document.getElementById("clear-history-btn");
  btn.disabled = true;
  btn.textContent = "Clearing…";

  try {
    const resp = await fetch("/history", { method: "DELETE" });
    const data = await resp.json();
    await loadHistory();
    alert(`Cleared ${data.deleted_entries} history item(s).`);
  } catch (err) {
    alert(`Error clearing history: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "Clear history";
  }
});

loadPapers();
loadHistory();
