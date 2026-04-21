function toast(message, level = "info") {
  const area = document.getElementById("toastArea");
  if (!area) return;

  const el = document.createElement("div");
  el.className = `toast align-items-center text-bg-${level} border-0`;
  el.role = "alert";
  el.ariaLive = "assertive";
  el.ariaAtomic = "true";

  el.innerHTML = `
    <div class="d-flex">
      <div class="toast-body">${escapeHtml(message)}</div>
      <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
    </div>
  `;

  area.appendChild(el);
  const t = new bootstrap.Toast(el, { delay: 4200 });
  t.show();

  el.addEventListener("hidden.bs.toast", () => el.remove());
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}