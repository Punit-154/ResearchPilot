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

  setLoading(true, "Retrieving evidence and generating answer…");

  try {
    const fullResp = await fetch(`/answer?q=${encodeURIComponent(question)}&${paperIdParams}`);
    const fullData = await fullResp.json();

    if (!fullResp.ok) {
      throw new Error(fullData.detail || "Request failed");
    }

    renderFullArchitectureResult(fullData);

    if (compareBaseline) {
      setLoading(true, "Running baseline for comparison…");
      // Baseline currently supports a single paper_id param per call; if
      // multiple papers are selected, run baseline against the first one
      // (baseline was never designed for multi-paper comparison).
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
    resultsArea.innerHTML = `<div class="error-box">Error: ${escapeHtml(err.message)}</div>`;
  } finally {
    setLoading(false);
  }
}

function renderFullArchitectureResult(data) {
  const resultsArea = document.getElementById("results-area");

  const block = document.createElement("div");
  block.className = "result-block full-arch";

  const claimsHtml = (data.claims || [])
    .map((claim) => {
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
          </div>
        </div>`;
    })
    .join("");

  const evidenceHtml = (data.evidence_used || [])
    .map((e) => `<span class="evidence-tag">[${e.label}] ${escapeHtml(e.paper_title)} p.${e.page}</span>`)
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

loadPapers();
