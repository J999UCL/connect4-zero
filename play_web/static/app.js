import * as THREE from "https://unpkg.com/three@0.160.0/build/three.module.js";
import { OrbitControls } from "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js";

const BOARD_SIZE = 4;
const SPACING = 1.25;
const HUMAN_COLOR = 0x48a7ff;
const BOT_COLOR = 0xef5d59;
const EMPTY_COLOR = 0x2c3942;
const OPEN_COLOR = 0x34c17b;
const BLOCKED_COLOR = 0x556068;

const canvas = document.querySelector("#board-canvas");
const statusText = document.querySelector("#status-text");
const detailText = document.querySelector("#detail-text");
const actionGrid = document.querySelector("#action-grid");
const resetButton = document.querySelector("#reset-button");
const lastHuman = document.querySelector("#last-human");
const lastBot = document.querySelector("#last-bot");
const moveCount = document.querySelector("#move-count");
const deviceText = document.querySelector("#device");

let state = null;
let busy = false;
let hoveredAction = null;
const actionButtons = [];
const columnPads = [];

const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  alpha: false,
  preserveDrawingBuffer: true,
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(0x0e1418, 1);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x0e1418, 10, 24);

const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
camera.position.set(5.8, 5.2, 7.4);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 0, 0);

const boardGroup = new THREE.Group();
scene.add(boardGroup);

const ambient = new THREE.AmbientLight(0xffffff, 0.7);
scene.add(ambient);

const keyLight = new THREE.DirectionalLight(0xffffff, 1.6);
keyLight.position.set(5, 9, 6);
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0x84b8ff, 0.45);
fillLight.position.set(-5, 4, -6);
scene.add(fillLight);

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

function actionToXY(action) {
  return [Math.floor(action / BOARD_SIZE), action % BOARD_SIZE];
}

function xyzToScene(x, y, z) {
  return new THREE.Vector3(
    (x - 1.5) * SPACING,
    z * SPACING + 0.42,
    (y - 1.5) * SPACING,
  );
}

function makeMaterial(color, opacity = 1) {
  return new THREE.MeshStandardMaterial({
    color,
    roughness: 0.55,
    metalness: 0.08,
    transparent: opacity < 1,
    opacity,
  });
}

const humanMaterial = makeMaterial(HUMAN_COLOR);
const botMaterial = makeMaterial(BOT_COLOR);
const emptyMaterial = makeMaterial(EMPTY_COLOR, 0.2);
const rodMaterial = makeMaterial(0x9fb1ac, 0.22);
const openPadMaterial = makeMaterial(OPEN_COLOR, 0.35);
const blockedPadMaterial = makeMaterial(BLOCKED_COLOR, 0.22);
const hoverPadMaterial = makeMaterial(0xf0c35b, 0.55);

function createActionButtons() {
  actionGrid.innerHTML = "";
  for (let action = 0; action < 16; action += 1) {
    const [x, y] = actionToXY(action);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "action-button";
    button.innerHTML = `<strong>${action}</strong><span>x=${x}, y=${y}</span>`;
    button.addEventListener("mouseenter", () => {
      hoveredAction = action;
      updateActionHighlights();
    });
    button.addEventListener("mouseleave", () => {
      hoveredAction = null;
      updateActionHighlights();
    });
    button.addEventListener("click", () => playMove(action));
    actionGrid.appendChild(button);
    actionButtons.push(button);
  }
}

function rebuildBoard() {
  boardGroup.clear();
  columnPads.length = 0;

  const baseGeometry = new THREE.BoxGeometry(BOARD_SIZE * SPACING + 0.7, 0.08, BOARD_SIZE * SPACING + 0.7);
  const base = new THREE.Mesh(baseGeometry, makeMaterial(0x19242b, 0.86));
  base.position.y = -0.15;
  boardGroup.add(base);

  const sphereGeometry = new THREE.SphereGeometry(0.33, 32, 20);
  const ghostGeometry = new THREE.SphereGeometry(0.12, 16, 10);
  const rodGeometry = new THREE.CylinderGeometry(0.025, 0.025, SPACING * 3.55, 10);
  const padGeometry = new THREE.BoxGeometry(0.9, 0.045, 0.9);

  for (let x = 0; x < BOARD_SIZE; x += 1) {
    for (let y = 0; y < BOARD_SIZE; y += 1) {
      const action = x * BOARD_SIZE + y;
      const pos = xyzToScene(x, y, 1.5);

      const rod = new THREE.Mesh(rodGeometry, rodMaterial);
      rod.position.set(pos.x, 2.28, pos.z);
      boardGroup.add(rod);

      const pad = new THREE.Mesh(padGeometry, openPadMaterial);
      pad.position.set(pos.x, -0.08, pos.z);
      pad.userData.action = action;
      columnPads.push(pad);
      boardGroup.add(pad);

      for (let z = 0; z < BOARD_SIZE; z += 1) {
        const value = state?.board?.[x]?.[y]?.[z] ?? 0;
        const material = value === 1 ? humanMaterial : value === -1 ? botMaterial : emptyMaterial;
        const geometry = value === 0 ? ghostGeometry : sphereGeometry;
        const stone = new THREE.Mesh(geometry, material);
        const stonePos = xyzToScene(x, y, z);
        stone.position.copy(stonePos);
        boardGroup.add(stone);
      }
    }
  }

  addLayerGuides();
  updateActionHighlights();
}

function addLayerGuides() {
  const material = new THREE.LineBasicMaterial({ color: 0x6f7f79, transparent: true, opacity: 0.22 });
  for (let z = 0; z < BOARD_SIZE; z += 1) {
    const y = z * SPACING + 0.42;
    const half = 1.5 * SPACING;
    const points = [
      new THREE.Vector3(-half, y, -half),
      new THREE.Vector3(half, y, -half),
      new THREE.Vector3(half, y, half),
      new THREE.Vector3(-half, y, half),
      new THREE.Vector3(-half, y, -half),
    ];
    boardGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), material));
  }
}

function updateActionHighlights() {
  const legal = state?.legalActions ?? Array(16).fill(false);
  actionButtons.forEach((button, action) => {
    button.disabled = busy || !legal[action];
    button.classList.toggle("active", hoveredAction === action);
  });

  columnPads.forEach((pad) => {
    const action = pad.userData.action;
    if (hoveredAction === action && legal[action] && !busy) {
      pad.material = hoverPadMaterial;
    } else if (legal[action] && !busy) {
      pad.material = openPadMaterial;
    } else {
      pad.material = blockedPadMaterial;
    }
  });
}

function updateText() {
  if (!state) return;
  statusText.textContent = state.message;
  detailText.textContent = state.winner
    ? `Winner: ${state.winner}`
    : `MCTS ${state.simulations} sims x ${state.rolloutBatchSize} rollouts`;
  lastHuman.textContent = state.lastHumanAction ?? "-";
  lastBot.textContent = state.lastBotAction ?? "-";
  moveCount.textContent = state.moveCount;
  deviceText.textContent = state.device;
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Request failed");
  }
  return payload;
}

async function loadState() {
  state = await fetchJson("/api/state");
  updateText();
  rebuildBoard();
}

async function playMove(action) {
  if (busy || !state?.legalActions?.[action]) return;
  busy = true;
  statusText.textContent = "Bot thinking...";
  updateActionHighlights();
  try {
    state = await fetchJson("/api/move", {
      method: "POST",
      body: JSON.stringify({ action }),
    });
  } catch (error) {
    statusText.textContent = error.message;
    await loadState();
    busy = false;
    return;
  }
  busy = false;
  updateText();
  rebuildBoard();
}

async function resetGame() {
  busy = true;
  updateActionHighlights();
  state = await fetchJson("/api/reset", { method: "POST", body: "{}" });
  busy = false;
  updateText();
  rebuildBoard();
}

function resize() {
  const rect = canvas.parentElement.getBoundingClientRect();
  renderer.setSize(rect.width, rect.height, false);
  camera.aspect = rect.width / rect.height;
  camera.updateProjectionMatrix();
}

function onPointerMove(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObjects(columnPads, false)[0];
  hoveredAction = hit ? hit.object.userData.action : null;
  updateActionHighlights();
}

function onPointerClick() {
  if (hoveredAction !== null) {
    playMove(hoveredAction);
  }
}

function animate() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

resetButton.addEventListener("click", resetGame);
renderer.domElement.addEventListener("pointermove", onPointerMove);
renderer.domElement.addEventListener("click", onPointerClick);
window.addEventListener("resize", resize);

createActionButtons();
resize();
loadState();
animate();
