(function () {
  const wrap = document.getElementById("messagesWrap");
  const btn = document.getElementById("addMsgBtn");
  if (!wrap || !btn) return;

  let count = wrap.querySelectorAll(".msg-block").length || 1;

  btn.addEventListener("click", () => {
    count += 1;
    const div = document.createElement("div");
    div.className = "border rounded p-3 bg-light msg-block";
    div.innerHTML = `
      <div class="d-flex justify-content-between align-items-center mb-2">
        <div class="fw-semibold small">Record ${count}</div>
        <button type="button" class="btn btn-sm btn-outline-danger removeBtn">Remove</button>
      </div>
      <div class="row g-2">
        <div class="col-md-4">
          <label class="form-label">Label</label>
          <input class="form-control" name="label[]" value="message_${count}" required>
        </div>
        <div class="col-md-8">
          <label class="form-label">Text</label>
          <input class="form-control" name="data[]" placeholder="Enter message text..." required>
        </div>
      </div>
      <div class="form-text">Each record becomes an independently labeled entry inside the packed container.</div>
    `;
    wrap.appendChild(div);

    div.querySelector(".removeBtn").addEventListener("click", () => {
      div.remove();
    });
  });
})();