const form = document.getElementById("query-form");
const queryInput = document.getElementById("query-input");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const resourceSummaryEl = document.getElementById("resource-summary");
const resourceListEl = document.getElementById("resource-list");

function resourceCount(value) {
  return Number.isFinite(Number(value)) ? Number(value) : 0;
}

function renderResourceSummary(payload) {
  if (!resourceSummaryEl) {
    return;
  }

  const packageStatusCounts = payload.package_status_counts || {};
  const missingComponentCounts = payload.missing_component_counts || {};
  const gatewayDeclaredResources = resourceCount(payload.gateway_declared_resources);
  const gatewayReadyResources = resourceCount(payload.gateway_ready_resources);

  const cards = [
    {
      label: "Enabled",
      value: `${resourceCount(payload.enabled_resources)} / ${resourceCount(payload.total_resources)}`,
      tone: "neutral",
    },
    {
      label: "Know-How",
      value: `${resourceCount(payload.resources_with_knowhow)} resources`,
      tone: "neutral",
    },
    {
      label: "Package Health",
      value: `ready ${resourceCount(packageStatusCounts.ready)} · degraded ${resourceCount(packageStatusCounts.degraded)}`,
      tone: resourceCount(packageStatusCounts.degraded) > 0 ? "warning" : "good",
    },
    {
      label: "Gateway",
      value: `${gatewayReadyResources} ready · ${gatewayDeclaredResources} declared`,
      tone: gatewayDeclaredResources > gatewayReadyResources ? "warning" : "good",
    },
  ];

  const missingComponents = Object.entries(missingComponentCounts)
    .map(([name, count]) => `${name}: ${count}`)
    .join(" · ");

  resourceSummaryEl.innerHTML = `
    <div class="resource-card-grid">
      ${cards.map((card) => `
        <article class="resource-card resource-card-${card.tone}">
          <span class="resource-card-label">${card.label}</span>
          <strong class="resource-card-value">${card.value}</strong>
        </article>
      `).join("")}
    </div>
    <p class="resource-meta">
      Missing components: ${missingComponents || "none"}
    </p>
  `;
}

function renderResourceList(payload) {
  if (!resourceListEl) {
    return;
  }

  const resources = Array.isArray(payload.resources) ? [...payload.resources] : [];
  resources.sort((left, right) => {
    const leftRisk = Number(Boolean(left.missing_components?.length)) + Number(left.package_status !== "ready") + Number(Boolean(left.gateway_declared));
    const rightRisk = Number(Boolean(right.missing_components?.length)) + Number(right.package_status !== "ready") + Number(Boolean(right.gateway_declared));
    if (leftRisk !== rightRisk) {
      return rightRisk - leftRisk;
    }
    return String(left.name || "").localeCompare(String(right.name || ""));
  });

  const rows = resources.map((resource) => {
    const missingComponents = Array.isArray(resource.missing_components) && resource.missing_components.length
      ? resource.missing_components.join(", ")
      : "none";
    const gatewayBits = resource.gateway_declared
      ? `${resource.gateway_transport || "gateway"} · ${resource.gateway_status || "unknown"}`
      : "not declared";
    const gatewayNamespace = resource.gateway_tool_namespace
      ? `<div class="resource-row-detail">namespace: ${resource.gateway_tool_namespace}</div>`
      : "";
    const gatewayMissingEnv = Array.isArray(resource.gateway_missing_env) && resource.gateway_missing_env.length
      ? `<div class="resource-row-detail">missing env: ${resource.gateway_missing_env.join(", ")}</div>`
      : "";
    return `
      <article class="resource-row">
        <div class="resource-row-top">
          <strong>${resource.name || "Unknown resource"}</strong>
          <span class="resource-pill">${resource.category || "unknown"}</span>
        </div>
        <div class="resource-row-detail">status: ${resource.status || "unknown"} · package: ${resource.package_status || "unknown"}</div>
        <div class="resource-row-detail">missing: ${missingComponents}</div>
        <div class="resource-row-detail">gateway: ${gatewayBits}</div>
        ${gatewayNamespace}
        ${gatewayMissingEnv}
      </article>
    `;
  });

  resourceListEl.innerHTML = rows.join("") || '<p class="resource-empty">No resources available.</p>';
}

async function loadResources() {
  if (!resourceSummaryEl || !resourceListEl) {
    return;
  }

  resourceSummaryEl.innerHTML = '<p class="resource-empty">Loading registry summary...</p>';
  resourceListEl.innerHTML = "";

  try {
    const response = await fetch("/resources");
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "resource request failed");
    }
    renderResourceSummary(payload);
    renderResourceList(payload);
  } catch (error) {
    const message = `Resource registry unavailable: ${error.message}`;
    resourceSummaryEl.innerHTML = `<p class="resource-empty">${message}</p>`;
    resourceListEl.innerHTML = "";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  statusEl.textContent = "Running...";
  resultEl.textContent = "";

  try {
    const response = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: queryInput.value,
        mode: "simple",
        resource_filter: [],
        save_md_report: true,
      }),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "request failed");
    }

    resultEl.textContent = [
      `Answer:\n${payload.answer || ""}`,
      payload.normalized_query ? `\nNormalized Query:\n${payload.normalized_query}` : "",
      payload.query_id ? `\nQuery ID:\n${payload.query_id}` : "",
    ].filter(Boolean).join("\n");
    await loadResources();
    statusEl.textContent = "Done";
  } catch (error) {
    resultEl.textContent = `Error: ${error.message}`;
    statusEl.textContent = "Failed";
  }
});

void loadResources();
