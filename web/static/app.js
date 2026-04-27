// Battleship vs AI — vanilla JS client.
//
// Two boards, alternating turns. /api/state on load; /api/action POSTs human
// move and (if game continues) gets the AI's response in the same round-trip.

const STATE_UNKNOWN = 0;
const STATE_MISS = 1;
const STATE_HIT = 2;
const STATE_SUNK = 3;

const COLS = "ABCDEFGHIJ";
const SIZE = 10;

let state = null;
let actionMode = "fire";
let lastHumanAction = null;
let lastAiAction = null;

// ----------------------------------------------------------------------
// Network
// ----------------------------------------------------------------------
async function api(path, opts = {}) {
  const r = await fetch(path, {
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  let body = null;
  try { body = await r.json(); } catch (_) {}
  if (!r.ok) {
    const msg = body && (body.detail || body.message) || r.statusText;
    throw new Error(msg);
  }
  return body;
}

async function loadState() {
  const r = await api("/api/state");
  state = r.state;
  render();
}

async function newGame() {
  const r = await api("/api/new_game", { method: "POST" });
  state = r.state;
  lastHumanAction = lastAiAction = null;
  render();
}

async function shuffle() {
  try {
    const r = await api("/api/shuffle", { method: "POST" });
    state = r.state;
    render();
  } catch (e) {
    flashStatus(e.message, "lose");
  }
}

async function act(kind, row, col) {
  try {
    const r = await api("/api/action", {
      method: "POST",
      body: JSON.stringify({ kind, row, col }),
    });
    state = r.state;
    lastHumanAction = r.human_action;
    lastAiAction = r.ai_action;
    render();
  } catch (e) {
    flashStatus(e.message, "lose");
  }
}

// ----------------------------------------------------------------------
// Rendering
// ----------------------------------------------------------------------
function render() {
  if (!state) return;
  renderBoard("human-board", state.human_board, /*isHumanFleet=*/true,
              lastAiAction);
  renderBoard("ai-board", state.ai_board, /*isHumanFleet=*/false,
              lastHumanAction);
  renderStatus();
  renderHistory();
}

function cellClasses(boardData, r, c, isHumanFleet) {
  const v = boardData.shot_board[r][c];
  const cleared = boardData.sonar_cleared[r][c];
  const yesOverlap = boardData.sonar_yes_overlap[r][c] > 0;
  const trueBoard = boardData.true_board;
  const hasShipHere = trueBoard ? trueBoard[r][c] > 0 : false;

  const classes = ["cell"];

  // Base color: shot result > sonar cleared > ship/water > fog
  if (v === STATE_SUNK)        classes.push("sunk");
  else if (v === STATE_HIT)    classes.push("hit");
  else if (v === STATE_MISS)   classes.push("miss");
  else if (cleared)            classes.push("cleared");
  else if (isHumanFleet) {
    classes.push(hasShipHere ? "ship" : "water");
  } else if (trueBoard) {
    // AI board after the game ends: reveal ships in cells we never shot
    classes.push(hasShipHere ? "ship" : "water");
  } else {
    classes.push("fog");
  }

  if (yesOverlap) classes.push("yes");
  return classes;
}

function renderBoard(elId, boardData, isHumanFleet, lastAction) {
  const el = document.getElementById(elId);
  el.innerHTML = "";

  // top-left empty corner
  el.appendChild(label(""));
  for (let c = 0; c < SIZE; c++) el.appendChild(label(COLS[c]));

  for (let r = 0; r < SIZE; r++) {
    el.appendChild(label(String(r + 1)));
    for (let c = 0; c < SIZE; c++) {
      const cell = document.createElement("div");
      cell.className = cellClasses(boardData, r, c, isHumanFleet).join(" ");
      cell.dataset.row = r;
      cell.dataset.col = c;
      // Highlight the *most recent* opponent action on each board
      if (lastAction && lastAction.row === r && lastAction.col === c) {
        cell.classList.add("last");
      }
      if (!isHumanFleet) {
        cell.addEventListener("click", onAiCellClick);
      }
      el.appendChild(cell);
    }
  }

  const stats = document.getElementById(
    isHumanFleet ? "human-stats" : "ai-stats");
  const total = boardData.total_ship_cells;
  const sunkSide = isHumanFleet ? "AI" : "You";
  const turnSide = isHumanFleet ? "AI" : "You";
  stats.innerHTML = `
    <span><strong>${boardData.hits_scored}/${total}</strong> hits</span>
    <span>${turnSide} sonar: <strong>${boardData.sonar_charges_remaining}/${boardData.sonar_max}</strong></span>
  `;

  if (!isHumanFleet) {
    const aiBoard = document.getElementById("ai-board");
    const clickable = state && !state.winner && state.turn === "human";
    aiBoard.classList.toggle("clickable", clickable);
  }
}

function label(text) {
  const el = document.createElement("div");
  el.className = "label";
  el.textContent = text;
  return el;
}

function onAiCellClick(e) {
  if (!state || state.winner || state.turn !== "human") return;
  const r = parseInt(e.currentTarget.dataset.row);
  const c = parseInt(e.currentTarget.dataset.col);

  if (actionMode === "fire") {
    const v = state.ai_board.shot_board[r][c];
    if (v !== STATE_UNKNOWN) {
      flashStatus("That cell has already been fired at.", "lose");
      return;
    }
    act("fire", r, c);
  } else {
    if (state.ai_board.sonar_charges_remaining <= 0) {
      flashStatus("No sonar charges remaining.", "lose");
      return;
    }
    act("sonar", r, c);
  }
}

// ----------------------------------------------------------------------
// Status / history
// ----------------------------------------------------------------------
function renderStatus() {
  const el = document.getElementById("status");
  el.classList.remove("win", "lose");
  const caption = document.getElementById("ai-caption");

  if (state.winner === "human") {
    el.textContent = `🏆  You won! Sunk the AI's fleet in ${state.ai_board.shots_fired} shots.`;
    el.classList.add("win");
    caption.textContent = "Game over.";
  } else if (state.winner === "ai") {
    el.textContent = `💀  AI won. It sunk your fleet in ${state.human_board.shots_fired} shots.`;
    el.classList.add("lose");
    caption.textContent = "Game over.";
  } else if (state.turn === "human") {
    const lastBit = lastAiAction ? `  (Last AI: ${formatAction(lastAiAction)})` : "";
    el.textContent = `Your turn — ${actionMode === "sonar" ? "sonar (3×3 query)" : "fire"}.${lastBit}`;
    caption.textContent = "Click a cell to fire";
  } else {
    el.textContent = "AI thinking…";
    caption.textContent = "AI's turn";
  }

  // Disable shuffle once the game has started
  document.getElementById("shuffle-btn").disabled =
    !state || state.started || state.winner;

  // If the player is out of sonar charges, snap mode back to fire
  if (state.ai_board.sonar_charges_remaining <= 0 && actionMode === "sonar") {
    document.querySelector('input[name=mode][value=fire]').checked = true;
    actionMode = "fire";
  }
  document.querySelector('input[name=mode][value=sonar]').disabled =
    state.ai_board.sonar_charges_remaining <= 0;
}

function flashStatus(msg, cls) {
  const el = document.getElementById("status");
  el.classList.remove("win", "lose");
  el.classList.add(cls || "lose");
  el.textContent = msg;
  setTimeout(() => renderStatus(), 1800);
}

function formatAction(a) {
  if (!a) return "";
  const cell = `${COLS[a.col]}${a.row + 1}`;
  if (a.kind === "fire") {
    if (a.result.sunk)      return `FIRE ${cell} → SUNK (${a.result.sunk_ship_size})`;
    if (a.result.hit)       return `FIRE ${cell} → HIT`;
    return `FIRE ${cell} → miss`;
  }
  return `SONAR ${cell} → ${a.result.sonar_hit ? "ship in zone" : "all clear"}`;
}

function renderHistory() {
  const list = document.getElementById("history");
  list.innerHTML = "";
  for (const a of state.history) {
    const li = document.createElement("li");
    li.className = a.side;
    li.textContent = `${a.side === "human" ? "You" : "AI"}: ${formatAction(a)}`;
    list.appendChild(li);
  }
  document.getElementById("history-count").textContent =
    state.history.length ? `(${state.history.length})` : "";
  list.scrollTop = list.scrollHeight;
}

// ----------------------------------------------------------------------
// Wire up
// ----------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("shuffle-btn").addEventListener("click", shuffle);
  document.getElementById("new-btn").addEventListener("click", newGame);
  document.querySelectorAll('input[name=mode]').forEach(el => {
    el.addEventListener("change", e => {
      actionMode = e.target.value;
      renderStatus();
    });
  });
  loadState();
});
