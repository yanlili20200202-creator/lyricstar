/*jshint esversion: 11 */
/*
  Transformers.js docs:
  https://huggingface.co/docs/transformers.js/index
  Embedding model:
  https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX
  UMAP-JS must be included in index.html so UMAP.UMAP exists (global).
*/

let autoTokenizer, autoModel, matmul;
let loaded = false;

let lines = [];
let points = [];
let corpusEmbeddings = [];
let umapRaw2D = null;

let inputField, button;
let resultText = "Loading model...";
let hoverText = "";

let topMatches = [];
let simMin = 0, simMax = 1;
let rankOf = [];

/* ----------------------- TUNE ----------------------- */
const BACKGROUND_SONG_COUNT = 600;
const TOP_K = 25;

const UMAP_NEIGHBORS = 45;
const UMAP_MINDIST = 0.35;

const JITTER = 22;
const DRIFT = 0.75;
const PARALLAX = 26;

const SIZE_MIN = 0.9;
const SIZE_MAX = 12.5;
const GLOW_MULT = 5.0;

const HAZE_ALPHA = 10;
const HAZE_SIZE = 2.3;

const RELAX_ITERS = 22;
const RELAX_RADIUS = 28;
const RELAX_STRENGTH = 0.06;

const PAD = 14;
const BOTTOM_UI_H = 170;

/* ----------------------- CAMERA ----------------------- */
let camZoom = 1.0, camZoomTarget = 1.0;
let focusX = 0, focusY = 0, focusXTarget = 0, focusYTarget = 0;
const ZOOM_MIN = 1.0, ZOOM_MAX = 2.8;
const CAM_LERP = 0.12;
const WHEEL_ZOOM_SPEED = 0.0016;

/* ----------------------- CANVAS DOM ----------------------- */
let cnv;

function preload() {
  lines = loadStrings("data_ai.txt");
}

function setup() {
  cnv = createCanvas(windowWidth, windowHeight);
  pixelDensity(1);
  textFont("monospace");
  textSize(12);

  if (lines.length > BACKGROUND_SONG_COUNT) lines = lines.slice(0, BACKGROUND_SONG_COUNT);

  // --- DOM UI: attach to canvas parent + absolute (stable in p5 editor) ---
  inputField = createElement("textarea");
  inputField.attribute("rows", "2");
  inputField.attribute("placeholder", "Type a feeling / scene / genre...");
  styleInput(inputField);

  button = createButton("Search");
  styleButton(button);
  button.attribute("disabled", true);
  button.mousePressed(runSearch);

  const parentEl = cnv.elt.parentElement;
  parentEl.style.position = "relative";
  parentEl.appendChild(inputField.elt);
  parentEl.appendChild(button.elt);

  layoutUI();
  loadTransformers();
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  layoutUI();
  if (umapRaw2D && points.length) {
    remapPointsToCanvas();
    resetCameraToCenter();
  }
}

/* ----------------------- UI: always centered ----------------------- */
function layoutUI() {
  const w = Math.min(720, Math.max(280, width * 0.6));

  // center horizontally, near bottom
  const top = height - BOTTOM_UI_H + 30;

  inputField.elt.style.position = "absolute";
  inputField.elt.style.width = w + "px";
  inputField.elt.style.left = "50%";
  inputField.elt.style.top = top + "px";
  inputField.elt.style.transform = "translateX(-50%)";

  button.elt.style.position = "absolute";
  button.elt.style.left = "50%";
  button.elt.style.top = (top + 62) + "px";
  button.elt.style.transform = "translateX(-50%)";
}

function styleInput(el) {
  el.elt.style.resize = "none";
  el.elt.style.padding = "10px";
  el.elt.style.borderRadius = "12px";
  el.elt.style.border = "1px solid rgba(255,255,255,0.15)";
  el.elt.style.background = "rgba(0,0,0,0.55)";
  el.elt.style.color = "white";
  el.elt.style.outline = "none";
  el.elt.style.fontFamily = "monospace";
  el.elt.style.zIndex = 9999;
  el.elt.style.boxSizing = "border-box";
}

function styleButton(el) {
  el.elt.style.padding = "8px 14px";
  el.elt.style.borderRadius = "12px";
  el.elt.style.border = "1px solid rgba(255,255,255,0.15)";
  el.elt.style.background = "rgba(255,255,255,0.08)";
  el.elt.style.color = "white";
  el.elt.style.fontFamily = "monospace";
  el.elt.style.cursor = "pointer";
  el.elt.style.zIndex = 9999;
}

/* ----------------------- wheel zoom ----------------------- */
function mouseWheel(event) {
  const factor = Math.pow(2, (-event.delta * WHEEL_ZOOM_SPEED));
  camZoomTarget = constrain(camZoomTarget * factor, ZOOM_MIN, ZOOM_MAX);
  return false;
}

/* ----------------------- DRAW ----------------------- */
function draw() {
  background(6);
  drawHeader();

  if (!loaded) {
    fill(180);
    textAlign(LEFT, TOP);
    text("Tip: first load may take a while (WebGPU).", PAD, PAD + 50);
    return;
  }

  camZoom = lerp(camZoom, camZoomTarget, CAM_LERP);
  focusX = lerp(focusX, focusXTarget, CAM_LERP);
  focusY = lerp(focusY, focusYTarget, CAM_LERP);

  // ✅ 没搜索也画星云：hasSearch=false 时 t=0.22 -> pop 有基础亮度
  drawPointCloud();
  drawMatchesPanel();
  drawHover();
}

function drawHeader() {
  noStroke();
  fill(0, 150);
  rect(PAD, PAD, width - PAD * 2, 64, 14);

  fill(245);
  textAlign(LEFT, TOP);
  text(resultText, PAD + 12, PAD + 10);

  fill(190);
  text(`Songs in cloud: ${lines.length}`, PAD + 12, PAD + 36);
}

function colorByPop(pop) {
  const p = constrain(pop, 0, 1);
  const r = lerp(40, 255, pow(p, 0.85));
  const g = lerp(60, 240, pow(p, 1.15));
  const b = lerp(160, 40, pow(p, 0.95));
  return [r, g, b];
}

function drawPointCloud() {
  hoverText = "";

  const mx = map(mouseX, 0, width, -1, 1);
  const my = map(mouseY, 0, height, -1, 1);

  const cx = (PAD + (width - PAD)) * 0.5;
  const cy = ((PAD + 80) + (height - PAD - BOTTOM_UI_H)) * 0.5;

  // haze base layer (fills canvas like nebula)
  noStroke();
  fill(140, 170, 255, HAZE_ALPHA);
  for (const p of points) {
    const par = 1 - p.z;
    const baseX = cx + (p.bx - focusX) * camZoom;
    const baseY = cy + (p.by - focusY) * camZoom;
    circle(baseX + mx * PARALLAX * par, baseY + my * PARALLAX * par, HAZE_SIZE);
  }

  const sortedIdx = points.map((_, i) => i).sort((a, b) => points[a].z - points[b].z);
  const hasSearch = (simMax > simMin + 1e-9);

  for (let order = 0; order < sortedIdx.length; order++) {
    const i = sortedIdx[order];
    const p = points[i];

    // ✅ no search -> base t gives nebula look
    let t = 0.22;
    if (hasSearch) {
      t = (p.similarity - simMin) / (simMax - simMin + 1e-9);
      t = constrain(t, 0, 1);
    }
    const pop = pow(t, 2.9);

    const rnk = (rankOf && rankOf.length) ? rankOf[i] : 9999;
    const rankFactor = (rnk < 120) ? map(rnk, 0, 119, 1.45, 0.55) : 0.5;

    const jx = (noise(p.seed + 1000) - 0.5) * JITTER;
    const jy = (noise(p.seed + 2000) - 0.5) * JITTER;
    const dx = (noise(p.seed + frameCount * 0.004) - 0.5) * DRIFT;
    const dy = (noise(p.seed + 999 + frameCount * 0.004) - 0.5) * DRIFT;

    const par = 1 - p.z;
    const baseX = cx + (p.bx - focusX) * camZoom;
    const baseY = cy + (p.by - focusY) * camZoom;

    const px = baseX + jx + dx + mx * PARALLAX * par;
    const py = baseY + jy + dy + my * PARALLAX * par;

    let size = constrain(lerp(SIZE_MIN, SIZE_MAX, pop) * rankFactor, SIZE_MIN, SIZE_MAX);
    const [r, g, b] = colorByPop(pop);
    const a = lerp(10, 250, pow(pop, 0.95));

    if (pop > 0.30) {
      fill(r, g, b, lerp(10, 170, (pop - 0.30) / 0.70));
      circle(px, py, size * GLOW_MULT);
    }
    fill(r, g, b, a);
    circle(px, py, size);

    if (dist(mouseX, mouseY, px, py) < max(7, size * 1.8)) hoverText = p.text;
  }
}

function drawHover() {
  if (!hoverText) return;
  const label = hoverText.length > 170 ? hoverText.slice(0, 167) + "..." : hoverText;

  noStroke();
  fill(0, 190);
  const w = min(width - 40, textWidth(label) + 20);
  rect(mouseX + 10, mouseY - 26, w, 24, 8);

  fill(255);
  textAlign(LEFT, CENTER);
  text(label, mouseX + 18, mouseY - 14);
}

function drawMatchesPanel() {
  if (!loaded) return;

  // keep panel on right, above centered input
  const panelW = min(680, width - 2 * PAD);
  const panelX = width - panelW - PAD;
  const panelY = height - BOTTOM_UI_H + 6;
  const panelH = 120;

  noStroke();
  fill(0, 150);
  rect(panelX, panelY, panelW, panelH, 14);

  fill(255);
  textAlign(LEFT, TOP);
  text("Top matches:", panelX + 12, panelY + 10);

  fill(220);
  let y = panelY + 30;
  for (let i = 0; i < topMatches.length; i++) {
    const m = topMatches[i];
    let line = `#${i + 1} (${m.score.toFixed(3)}) ${m.text}`;
    if (line.length > 140) line = line.slice(0, 137) + "...";
    text(line, panelX + 12, y);
    y += 12;
    if (y > panelY + panelH - 10) break;
  }
}

/* ----------------------- SEARCH ----------------------- */
async function runSearch() {
  if (!autoModel) return;
  const query = inputField.value().trim();
  if (!query) return;

  resultText = "Thinking…";

  const prefixes = {
    query: "task: search result | query: ",
    line: "title: none | text: "
  };

  const allTexts = [prefixes.query + query, ...lines.map(s => prefixes.line + s)];
  const inputs = await autoTokenizer(allTexts, { padding: true, truncation: true });
  const output = await autoModel(inputs);

  const allEmbeddings = output.sentence_embedding;
  const scores = await matmul(allEmbeddings, allEmbeddings.transpose(1, 0));
  const sims = scores.tolist()[0].slice(1);

  simMin = Infinity; simMax = -Infinity;
  for (let i = 0; i < points.length; i++) {
    const s = sims[i];
    points[i].similarity = s;
    simMin = min(simMin, s);
    simMax = max(simMax, s);
  }

  const idxs = Array.from({ length: sims.length }, (_, i) => i).sort((a, b) => sims[b] - sims[a]);
  rankOf = new Array(points.length);
  for (let r = 0; r < idxs.length; r++) rankOf[idxs[r]] = r;

  topMatches = idxs.slice(0, TOP_K).map(i => ({ idx: i, score: sims[i], text: lines[i] }));

  const best = topMatches[0];
  resultText = `Best match:\n${best.text}\nScore: ${best.score.toFixed(3)}`;

  computeFocusAndZoom(sims, idxs);
}

function computeFocusAndZoom(sims, idxs) {
  const N = min(120, idxs.length);
  let sx = 0, sy = 0, sw = 0;

  for (let k = 0; k < N; k++) {
    const i = idxs[k];
    const t = (sims[i] - simMin) / (simMax - simMin + 1e-9);
    const w = pow(constrain(t, 0, 1), 3.2) + 0.0001;
    sx += points[i].bx * w;
    sy += points[i].by * w;
    sw += w;
  }

  const fx = sx / sw;
  const fy = sy / sw;

  let rad = 0;
  for (let k = 0; k < N; k++) {
    const i = idxs[k];
    const dx = points[i].bx - fx;
    const dy = points[i].by - fy;
    rad += sqrt(dx * dx + dy * dy);
  }
  rad /= max(1, N);

  const viewW = width - PAD * 2;
  const viewH = height - PAD - (PAD + 80) - BOTTOM_UI_H;
  const viewR = 0.22 * min(viewW, viewH);

  camZoomTarget = constrain(viewR / max(1e-6, rad), ZOOM_MIN, ZOOM_MAX);
  focusXTarget = fx;
  focusYTarget = fy;
}

function resetCameraToCenter() {
  camZoomTarget = 1.0;

  let mx = 0, my = 0;
  for (const p of points) { mx += p.bx; my += p.by; }
  mx /= max(1, points.length);
  my /= max(1, points.length);

  focusXTarget = mx;
  focusYTarget = my;

  focusX = focusXTarget;
  focusY = focusYTarget;
  camZoom = camZoomTarget;
}

/* ----------------------- UMAP ----------------------- */
async function computeCorpusUMAP() {
  const formatted = lines.map(s => "title: none | text: " + s);
  const inputs = await autoTokenizer(formatted, { padding: true, truncation: true });
  const output = await autoModel(inputs);
  corpusEmbeddings = output.sentence_embedding.tolist();

  const umap = new UMAP.UMAP({
    nComponents: 2,
    nNeighbors: UMAP_NEIGHBORS,
    minDist: UMAP_MINDIST
  });
  umapRaw2D = umap.fit(corpusEmbeddings);

  points = [];
  for (let i = 0; i < lines.length; i++) {
    points.push({
      rx: umapRaw2D[i][0],
      ry: umapRaw2D[i][1],
      bx: 0,
      by: 0,
      similarity: 0,
      text: lines[i],
      seed: random(10000),
      z: random(0.06, 1.0)
    });
  }

  remapPointsToCanvas();
  resetCameraToCenter();
}

function remapPointsToCanvas() {
  let arr = points.map(p => ({ x: p.rx, y: p.ry }));
  arr = pcaRotate2D(arr);

  const fitted = fitToRect(arr, {
    left: PAD, right: width - PAD,
    top: PAD + 80, bottom: height - PAD - BOTTOM_UI_H,
    pad: 0.03
  });

  for (let i = 0; i < points.length; i++) {
    points[i].bx = fitted[i].x;
    points[i].by = fitted[i].y;
  }

  relaxPoints(points, RELAX_ITERS, RELAX_RADIUS, RELAX_STRENGTH);

  const arr2 = points.map(p => ({ x: p.bx, y: p.by }));
  const fitted2 = fitToRect(arr2, {
    left: PAD, right: width - PAD,
    top: PAD + 80, bottom: height - PAD - BOTTOM_UI_H,
    pad: 0.03
  });

  for (let i = 0; i < points.length; i++) {
    points[i].bx = fitted2[i].x;
    points[i].by = fitted2[i].y;
  }
}

/* ----------------------- MATH ----------------------- */
function pcaRotate2D(arr) {
  let mx = 0, my = 0;
  for (const p of arr) { mx += p.x; my += p.y; }
  mx /= arr.length; my /= arr.length;

  let sxx = 0, syy = 0, sxy = 0;
  for (const p of arr) {
    const x = p.x - mx, y = p.y - my;
    sxx += x * x; syy += y * y; sxy += x * y;
  }
  sxx /= arr.length; syy /= arr.length; sxy /= arr.length;

  const theta = 0.5 * Math.atan2(2 * sxy, (sxx - syy));
  const ct = Math.cos(-theta), st = Math.sin(-theta);

  return arr.map(p => {
    const x0 = p.x - mx, y0 = p.y - my;
    return { x: x0 * ct - y0 * st, y: x0 * st + y0 * ct };
  });
}

function fitToRect(arr, { left, right, top, bottom, pad = 0.0 }) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of arr) {
    minX = min(minX, p.x); maxX = max(maxX, p.x);
    minY = min(minY, p.y); maxY = max(maxY, p.y);
  }
  const w = max(1e-9, maxX - minX);
  const h = max(1e-9, maxY - minY);

  const targetW = (right - left) * (1 - pad);
  const targetH = (bottom - top) * (1 - pad);
  const scale = min(targetW / w, targetH / h);

  const cx = (minX + maxX) * 0.5;
  const cy = (minY + maxY) * 0.5;

  return arr.map(p => ({
    x: (left + right) * 0.5 + (p.x - cx) * scale,
    y: (top + bottom) * 0.5 + (p.y - cy) * scale
  }));
}

function relaxPoints(pts, iters, radius, strength) {
  const r2 = radius * radius;
  for (let k = 0; k < iters; k++) {
    for (let i = 0; i < pts.length; i++) {
      let fx = 0, fy = 0;
      for (let j = 0; j < pts.length; j++) {
        if (i === j) continue;
        const dx = pts[i].bx - pts[j].bx;
        const dy = pts[i].by - pts[j].by;
        const d2 = dx * dx + dy * dy;
        if (d2 > 1e-6 && d2 < r2) {
          const d = Math.sqrt(d2);
          const push = (radius - d) / radius;
          fx += (dx / d) * push;
          fy += (dy / d) * push;
        }
      }
      pts[i].bx += fx * strength;
      pts[i].by += fy * strength;
    }
  }
}

/* ----------------------- MODEL ----------------------- */
async function loadTransformers() {
  const transformers = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers");

  autoTokenizer = await transformers.AutoTokenizer.from_pretrained(
    "onnx-community/embeddinggemma-300m-ONNX"
  );

  autoModel = await transformers.AutoModel.from_pretrained(
    "onnx-community/embeddinggemma-300m-ONNX",
    { device: "webgpu", dtype: "fp32" }
  );

  matmul = transformers.matmul;

  resultText = "Model loaded. Computing UMAP…";
  await computeCorpusUMAP();

  loaded = true;
  resultText = "Enter a query and click Search!";
  button.removeAttribute("disabled");
}