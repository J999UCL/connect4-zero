import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const sceneHost = document.querySelector("#scene");
const statusEl = document.querySelector("#status");
const plyEl = document.querySelector("#ply");
const rootValueEl = document.querySelector("#root-value");
const searchMsEl = document.querySelector("#search-ms");
const depthEl = document.querySelector("#depth");
const policyGridEl = document.querySelector("#policy-grid");
const moveListEl = document.querySelector("#move-list");
const resetButton = document.querySelector("#reset");

const spacing = 1.18;
const pieceRadius = 0.32;
const boardOffset = (3 * spacing) / 2;
const boardOrigin = new THREE.Vector3(0, 0.95, 0);
const boardBaseCenter = boardOrigin.clone();
let state = null;
let thinking = false;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x081012);

const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
camera.position.copy(boardBaseCenter).add(new THREE.Vector3(4.8, 5.8, 8.2));

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
sceneHost.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.copy(boardBaseCenter);
controls.minDistance = 5.0;
controls.maxDistance = 16;
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.68));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.7);
keyLight.position.set(4, 7, 6);
scene.add(keyLight);

const boardGroup = new THREE.Group();
const pieceGroup = new THREE.Group();
boardGroup.position.copy(boardOrigin);
pieceGroup.position.copy(boardOrigin);
scene.add(boardGroup);
scene.add(pieceGroup);

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const clickTargets = [];
const heatTiles = [];
let pointerDown = null;

const humanMaterial = new THREE.MeshStandardMaterial({
  color: 0x22d3ee,
  roughness: 0.38,
  metalness: 0.08,
});
const botMaterial = new THREE.MeshStandardMaterial({
  color: 0xf97316,
  roughness: 0.42,
  metalness: 0.08,
});
const ghostMaterial = new THREE.MeshBasicMaterial({
  color: 0xe5f4f2,
  transparent: true,
  opacity: 0.18,
});

function cellPosition(x, y, z = 0) {
  return new THREE.Vector3(x * spacing - boardOffset, z * spacing + pieceRadius, y * spacing - boardOffset);
}

function actionToXY(action) {
  return { x: action % 4, y: Math.floor(action / 4) };
}

function buildBoard() {
  const baseMaterial = new THREE.MeshStandardMaterial({
    color: 0x173231,
    roughness: 0.72,
    metalness: 0.02,
  });
  const edgeMaterial = new THREE.LineBasicMaterial({ color: 0x86efdf, transparent: true, opacity: 0.42 });
  const tileGeometry = new THREE.BoxGeometry(0.96, 0.06, 0.96);
  const heatGeometry = new THREE.BoxGeometry(0.98, 0.025, 0.98);
  const targetGeometry = new THREE.BoxGeometry(1.06, 0.08, 1.06);
  const slotGeometry = new THREE.TorusGeometry(pieceRadius * 1.08, 0.012, 8, 36);

  for (let action = 0; action < 16; action += 1) {
    const { x, y } = actionToXY(action);
    const base = cellPosition(x, y, 0);

    const tile = new THREE.Mesh(tileGeometry, baseMaterial);
    tile.position.set(base.x, -0.05, base.z);
    boardGroup.add(tile);

    const heatMaterial = new THREE.MeshBasicMaterial({
      color: 0x2dd4bf,
      transparent: true,
      opacity: 0,
      depthWrite: false,
    });
    const heat = new THREE.Mesh(heatGeometry, heatMaterial);
    heat.position.set(base.x, 0.015, base.z);
    boardGroup.add(heat);
    heatTiles[action] = heat;

    const target = new THREE.Mesh(targetGeometry, new THREE.MeshBasicMaterial({ visible: false }));
    target.position.set(base.x, 0.08, base.z);
    target.userData.action = action;
    boardGroup.add(target);
    clickTargets.push(target);

    for (let z = 0; z < 4; z += 1) {
      const slot = new THREE.Mesh(slotGeometry, ghostMaterial);
      slot.position.copy(cellPosition(x, y, z));
      slot.rotation.x = Math.PI / 2;
      boardGroup.add(slot);
    }
  }

  const size = spacing * 4;
  const grid = new THREE.GridHelper(size, 4, 0x9ce7dc, 0x2c4a48);
  grid.position.y = -0.015;
  boardGroup.add(grid);

  const axes = new THREE.Group();
  const linePoints = [
    new THREE.Vector3(-boardOffset - 0.55, 0, -boardOffset - 0.55),
    new THREE.Vector3(boardOffset + 0.55, 0, -boardOffset - 0.55),
    new THREE.Vector3(-boardOffset - 0.55, 0, -boardOffset - 0.55),
    new THREE.Vector3(-boardOffset - 0.55, 0, boardOffset + 0.55),
  ];
  for (let i = 0; i < linePoints.length; i += 2) {
    const geometry = new THREE.BufferGeometry().setFromPoints([linePoints[i], linePoints[i + 1]]);
    axes.add(new THREE.Line(geometry, edgeMaterial));
  }
  boardGroup.add(axes);
}

function updatePieces(nextState) {
  pieceGroup.clear();
  const geometry = new THREE.SphereGeometry(pieceRadius, 32, 18);
  for (const cell of nextState.cells) {
    const sphere = new THREE.Mesh(geometry, cell.owner === "human" ? humanMaterial : botMaterial);
    const pos = cellPosition(cell.x, cell.y, cell.z);
    sphere.position.copy(pos);
    pieceGroup.add(sphere);
  }
}

function updateHeat(nextState) {
  const policy = nextState.lastSearch?.policy ?? [];
  for (let action = 0; action < 16; action += 1) {
    const value = policy[action] ?? 0;
    const tile = heatTiles[action];
    tile.material.opacity = Math.min(0.78, value * 2.4);
    tile.material.color.setHSL(0.47 - value * 0.28, 0.72, 0.48);
  }
}

function policyPercent(value) {
  return `${(100 * value).toFixed(value >= 0.1 ? 0 : 1)}%`;
}

function updatePanel(nextState) {
  const search = nextState.lastSearch;
  plyEl.textContent = nextState.ply;
  rootValueEl.textContent = search ? search.rootValue.toFixed(3) : "-";
  searchMsEl.textContent = search ? `${search.searchMs.toFixed(0)} ms` : "-";
  depthEl.textContent = search ? search.maxDepth : "-";

  if (nextState.terminal) {
    statusEl.textContent = nextState.message;
  } else if (thinking) {
    statusEl.textContent = "Bot thinking...";
  } else if (nextState.humanToMove) {
    statusEl.textContent = "Your move. Click a base square to drop a piece.";
  } else {
    statusEl.textContent = "Bot to move.";
  }

  policyGridEl.replaceChildren();
  for (let row = 3; row >= 0; row -= 1) {
    for (let x = 0; x < 4; x += 1) {
      const action = row * 4 + x;
      const cell = document.createElement("div");
      cell.className = "policy-cell";
      const legal = (nextState.legalMask & (1 << action)) !== 0;
      const prob = search ? search.policy[action] : 0;
      cell.style.background = legal
        ? `rgba(45, 212, 191, ${Math.min(0.36, prob * 1.4)})`
        : "rgba(255,255,255,0.035)";
      cell.innerHTML = `<div class="action">${action}</div><div class="prob">${legal ? policyPercent(prob) : "full"}</div>`;
      policyGridEl.appendChild(cell);
    }
  }

  moveListEl.replaceChildren();
  const rows = [];
  if (search) {
    for (let action = 0; action < 16; action += 1) {
      rows.push({
        action,
        policy: search.policy[action],
        visits: search.visitCounts[action],
        q: search.qValues[action],
      });
    }
    rows.sort((a, b) => b.visits - a.visits || a.action - b.action);
  }
  for (const row of rows.slice(0, 8)) {
    const item = document.createElement("div");
    item.className = "move-row";
    item.innerHTML = `
      <span>${row.action}</span>
      <span class="bar"><span style="width:${Math.min(100, row.policy * 100)}%"></span></span>
      <span>${row.visits}</span>
      <span>${row.q.toFixed(2)}</span>
    `;
    moveListEl.appendChild(item);
  }
}

function renderState(nextState) {
  state = nextState;
  updatePieces(nextState);
  updateHeat(nextState);
  updatePanel(nextState);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error ?? `request failed: ${response.status}`);
  }
  return payload;
}

async function askBotIfNeeded() {
  if (!state || state.terminal || state.humanToMove) {
    return;
  }
  thinking = true;
  updatePanel(state);
  try {
    renderState(await api("/api/bot", { method: "POST", body: "{}" }));
  } catch (error) {
    statusEl.textContent = error.message;
  } finally {
    thinking = false;
    if (state) {
      updatePanel(state);
    }
  }
}

async function playAction(action) {
  if (!state || thinking || state.terminal || !state.humanToMove) {
    return;
  }
  if ((state.legalMask & (1 << action)) === 0) {
    return;
  }
  try {
    renderState(await api("/api/move", { method: "POST", body: JSON.stringify({ action }) }));
    await askBotIfNeeded();
  } catch (error) {
    statusEl.textContent = error.message;
  }
}

function onPointerDown(event) {
  pointerDown = { x: event.clientX, y: event.clientY };
}

function onPointerUp(event) {
  if (pointerDown) {
    const dx = event.clientX - pointerDown.x;
    const dy = event.clientY - pointerDown.y;
    pointerDown = null;
    if (Math.hypot(dx, dy) > 6) {
      return;
    }
  }
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(clickTargets, false);
  if (hits.length > 0) {
    playAction(hits[0].object.userData.action);
  }
}

function resize() {
  const rect = sceneHost.getBoundingClientRect();
  camera.aspect = Math.max(1, rect.width) / Math.max(1, rect.height);
  camera.updateProjectionMatrix();
  renderer.setSize(rect.width, rect.height);
}

function animate() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

resetButton.addEventListener("click", async () => {
  if (thinking) {
    return;
  }
  try {
    renderState(await api("/api/new", { method: "POST", body: "{}" }));
  } catch (error) {
    statusEl.textContent = error.message;
  }
});
renderer.domElement.addEventListener("pointerdown", onPointerDown);
renderer.domElement.addEventListener("pointerup", onPointerUp);
window.addEventListener("resize", resize);

buildBoard();
resize();
animate();
renderState(await api("/api/state"));
await askBotIfNeeded();
