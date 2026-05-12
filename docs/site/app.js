const sampleBoard = [
  [
    [1, -1, 0, 0],
    [0, 1, -1, 0],
    [-1, 0, 0, 1],
    [0, 0, 0, -1],
  ],
  [
    [-1, 1, 0, 0],
    [0, -1, 1, 0],
    [1, 0, 0, -1],
    [0, 0, 0, 1],
  ],
  [
    [0, 0, 0, 0],
    [0, 1, -1, 0],
    [0, -1, 1, 0],
    [0, 0, 0, 0],
  ],
  [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
  ],
];

const stageText = {
  select: {
    title: "1. Selection",
    body:
      "Start at the root and repeatedly choose the child with the highest UCB score until the search reaches a terminal node or a node with unexpanded legal moves.",
    bullets: [
      "Legal actions come from the engine mask.",
      "Visited children use exploitation plus exploration.",
      "Child values are negated when compared at the parent.",
    ],
  },
  expand: {
    title: "2. Expansion",
    body:
      "Clone the node state, apply one legal action, then ask NodeStore for the child. Today this creates a tree node; later it can reuse a Zobrist DAG node.",
    bullets: [
      "The caller's root state is never mutated.",
      "Terminal children get exact values.",
      "Non-terminal children are ready for evaluation.",
    ],
  },
  evaluate: {
    title: "3. Evaluation",
    body:
      "The current evaluator replicates the leaf into a rollout batch and plays random games until terminal. The returned scalar is from the leaf player-to-move perspective.",
    bullets: [
      "Runs on CPU by default.",
      "Can target MPS or CUDA when configured.",
      "Future neural-net inference plugs into the same evaluator boundary.",
    ],
  },
  backprop: {
    title: "4. Backpropagation",
    body:
      "Walk back along the selected path. Add the value to each node, increment visits, and flip the sign at every edge because the player-to-move alternates.",
    bullets: [
      "leaf value v is added to the leaf",
      "-v is added to the parent",
      "the root mean value becomes the search value",
    ],
  },
  policy: {
    title: "5. SearchResult",
    body:
      "After all simulations, root child visit counts become a length-16 policy target. Illegal actions remain zero, and q_values are reported from the root perspective.",
    bullets: [
      "visit_counts[action] stores child visits",
      "policy = visit_counts / total_visits",
      "root_value = root.value_sum / root.visits",
    ],
  },
};

let selectedAction = 5;

function classForValue(value) {
  if (value === 1) return "current";
  if (value === -1) return "opponent";
  return "empty";
}

function renderBoard() {
  const board = document.querySelector("#hero-board");
  if (!board) return;
  const selectedX = Math.floor(selectedAction / 4);
  const selectedY = selectedAction % 4;
  board.innerHTML = "";

  for (let z = 3; z >= 0; z -= 1) {
    const layer = document.createElement("div");
    layer.className = "board-layer";

    const label = document.createElement("div");
    label.className = "layer-label";
    label.textContent = `z=${z}`;
    layer.appendChild(label);

    for (let x = 0; x < 4; x += 1) {
      for (let y = 0; y < 4; y += 1) {
        const cell = document.createElement("div");
        cell.className = `cell ${classForValue(sampleBoard[z][x][y])}`;
        if (x === selectedX && y === selectedY) {
          cell.classList.add("highlight");
        }
        cell.title = `x=${x}, y=${y}, z=${z}`;
        layer.appendChild(cell);
      }
    }

    board.appendChild(layer);
  }
}

function renderActionGrid() {
  const grid = document.querySelector("#action-grid");
  if (!grid) return;
  grid.innerHTML = "";

  for (let action = 0; action < 16; action += 1) {
    const x = Math.floor(action / 4);
    const y = action % 4;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "action-cell";
    button.dataset.action = String(action);
    button.innerHTML = `<strong>${action}</strong><span>x=${x}, y=${y}</span>`;
    button.addEventListener("mouseenter", () => setAction(action));
    button.addEventListener("click", () => setAction(action));
    grid.appendChild(button);
  }
  updateActionGrid();
}

function setAction(action) {
  selectedAction = action;
  updateActionGrid();
  renderBoard();
  const readout = document.querySelector("#action-readout");
  if (readout) {
    const x = Math.floor(action / 4);
    const y = action % 4;
    readout.textContent = `Action ${action} selects column x=${x}, y=${y}. The engine drops the next +1 stone into the first open z slot in that column.`;
  }
}

function updateActionGrid() {
  document.querySelectorAll(".action-cell").forEach((button) => {
    button.classList.toggle("active", Number(button.dataset.action) === selectedAction);
  });
}

function renderStage(stage) {
  document.body.dataset.stage = stage;
  document.querySelectorAll(".stage-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.stage === stage);
  });

  const copy = document.querySelector("#stage-copy");
  const info = stageText[stage];
  if (!copy || !info) return;

  copy.innerHTML = `
    <h3>${info.title}</h3>
    <p>${info.body}</p>
    <ul>${info.bullets.map((item) => `<li>${item}</li>`).join("")}</ul>
  `;
}

function bindStages() {
  document.querySelectorAll(".stage-button").forEach((button) => {
    button.addEventListener("click", () => renderStage(button.dataset.stage));
  });
}

renderActionGrid();
renderBoard();
bindStages();
renderStage("select");
