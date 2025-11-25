async function postBlob(url, field, file) {
  const fd = new FormData();
  fd.append(field, file);
  const res = await fetch(url, { method: "POST", body: fd });
  if (!res.ok) throw new Error(await res.text());
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

/* ===========================
   ẢNH: SIGN & LANE
   =========================== */

document.getElementById("btn-sign-img").onclick = async () => {
  const file = document.getElementById("sign-img").files[0];
  const variant = document.getElementById("sign-var").value;
  const conf = document.getElementById("sign-conf").value;
  if (!file) return alert("Chọn ảnh trước");

  const btn = document.getElementById("btn-sign-img");
  const oldText = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Loading...";

  const url = `/infer/sign/image?variant=${variant}&conf=${conf}`;
  try {
    const obj = await postBlob(url, "image", file);
    document.getElementById("sign-out").src = obj;
  } catch (e) {
    alert("Sign image lỗi: " + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = oldText;
  }
};

document.getElementById("btn-lane-img").onclick = async () => {
  const file = document.getElementById("lane-img").files[0];
  const variant = document.getElementById("lane-var").value;
  const conf = document.getElementById("lane-conf").value;
  if (!file) return alert("Chọn ảnh trước");

  const btn = document.getElementById("btn-lane-img");
  const oldText = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Loading...";

  const url = `/infer/lane/image?variant=${variant}&conf=${conf}`;
  try {
    const obj = await postBlob(url, "image", file);
    document.getElementById("lane-out").src = obj;
  } catch (e) {
    alert("Lane image lỗi: " + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = oldText;
  }
};

/* ===========================
   VIDEO: SIGN & LANE
   =========================== */

document.getElementById("btn-sign-vid").onclick = async () => {
  const file = document.getElementById("sign-vid").files[0];
  const variant = document.getElementById("sign-vid-var").value;
  const conf = document.getElementById("sign-vid-conf").value;
  const stride = document.getElementById("sign-vid-stride").value;
  if (!file) return alert("Chọn video trước");

  const btn = document.getElementById("btn-sign-vid");
  const oldText = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Loading...";

  const url = `/infer/sign/video?variant=${variant}&conf=${conf}&stride=${stride}`;
  try {
    const obj = await postBlob(url, "video", file);
    const v = document.getElementById("sign-vid-out");
    v.src = obj;
    v.play();
  } catch (e) {
    alert("Sign video lỗi: " + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = oldText;
  }
};

document.getElementById("btn-lane-vid").onclick = async () => {
  const file = document.getElementById("lane-vid").files[0];
  const variant = document.getElementById("lane-vid-var").value;
  const conf = document.getElementById("lane-vid-conf").value;
  const stride = document.getElementById("lane-vid-stride").value;
  if (!file) return alert("Chọn video trước");

  const btn = document.getElementById("btn-lane-vid");
  const oldText = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Loading...";

  const url = `/infer/lane/video?variant=${variant}&conf=${conf}&stride=${stride}`;
  try {
    const obj = await postBlob(url, "video", file);
    const v = document.getElementById("lane-vid-out");
    v.src = obj;
    v.play();
  } catch (e) {
    alert("Lane video lỗi: " + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = oldText;
  }
};

/* ===========================
   WEBCAM: SIGN & LANE
   =========================== */

const camVideo = document.getElementById("cam-video");
const camOut   = document.getElementById("cam-out");
const camCanvas = document.getElementById("cam-canvas");

const camStartBtn = document.getElementById("cam-start");
const camStopBtn  = document.getElementById("cam-stop");

const camTaskSelect = document.getElementById("cam-task");
const camLaneVarWrap = document.getElementById("cam-lane-var-wrap");
const camLaneVarSelect = document.getElementById("cam-lane-var");
const camConfInput = document.getElementById("cam-conf");
const camStatusSpan = document.getElementById("cam-status");

let camStream = null;
let camIntervalId = null;
let camIsSending = false;

// Hiển thị/ẩn chọn lane variant tùy task
camTaskSelect.addEventListener("change", () => {
  if (camTaskSelect.value === "lane") {
    camLaneVarWrap.style.display = "inline-flex";
  } else {
    camLaneVarWrap.style.display = "none";
  }
});

// Start camera
camStartBtn.addEventListener("click", async () => {
  try {
    camStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 },
      audio: false,
    });

    camVideo.srcObject = camStream;
    camStartBtn.disabled = true;
    camStopBtn.disabled = false;
    camStatusSpan.textContent = "Status: camera started";

    startCamLoop();
  } catch (err) {
    console.error("Error starting camera:", err);
    camStatusSpan.textContent = "Status: cannot start camera (check permission)";
  }
});

// Stop camera
camStopBtn.addEventListener("click", () => {
  stopCamLoop();
  if (camStream) {
    camStream.getTracks().forEach((t) => t.stop());
    camStream = null;
  }
  camStartBtn.disabled = false;
  camStopBtn.disabled = true;
  camStatusSpan.textContent = "Status: stopped";
});

function startCamLoop() {
  if (camIntervalId != null) return;
  camIntervalId = setInterval(() => {
    if (!camStream || !camVideo.videoWidth || camIsSending) return;
    captureAndSendCamFrame();
  }, 300); // ~3 fps
}

function stopCamLoop() {
  if (camIntervalId != null) {
    clearInterval(camIntervalId);
    camIntervalId = null;
  }
}

async function captureAndSendCamFrame() {
  try {
    camIsSending = true;

    const w = camVideo.videoWidth;
    const h = camVideo.videoHeight;
    if (!w || !h) {
      camIsSending = false;
      return;
    }

    camCanvas.width = w;
    camCanvas.height = h;
    const ctx = camCanvas.getContext("2d");
    ctx.drawImage(camVideo, 0, 0, w, h);

    const blob = await new Promise((resolve) =>
      camCanvas.toBlob((b) => resolve(b), "image/jpeg", 0.8)
    );
    if (!blob) {
      camIsSending = false;
      return;
    }

    const task = camTaskSelect.value; // "lane" | "sign"
    const conf = parseFloat(camConfInput.value || "0.35") || 0.35;
    let url;
    let variant;

    if (task === "lane") {
      // Lane: dùng best/last (.pt) 
      variant = camLaneVarSelect.value;
      url = `/infer/lane/image?variant=${encodeURIComponent(variant)}&conf=${conf.toFixed(2)}`;
    } else {
      // Sign: luôn dùng best (onnx best)
      variant = "best";
      url = `/infer/sign/image?variant=${encodeURIComponent(variant)}&conf=${conf.toFixed(2)}`;
    }

    const fd = new FormData();
    fd.append("image", blob, "frame.jpg");

    const res = await fetch(url, { method: "POST", body: fd });
    if (!res.ok) {
      console.error("Webcam response not OK:", res.status, await res.text());
      camStatusSpan.textContent = `Status: error ${res.status}`;
      camIsSending = false;
      return;
    }

    const outBlob = await res.blob();
    const objUrl = URL.createObjectURL(outBlob);
    camOut.src = objUrl;

    camStatusSpan.textContent = `Status: OK (${task}, variant=${variant}, conf=${conf.toFixed(2)})`;
  } catch (err) {
    console.error("Webcam capture/send error:", err);
    camStatusSpan.textContent = "Status: error (xem console)";
  } finally {
    camIsSending = false;
  }
}
