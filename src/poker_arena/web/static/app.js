const state = {
  snapshot: null,
  socket: null,
  seatToken: localStorage.getItem("pokerArenaSeatToken"),
  hostToken: localStorage.getItem("pokerArenaHostToken") || "",
};

const params = new URLSearchParams(window.location.search);
if (params.get("room_code")) {
  document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("roomCodeInput").value = params.get("room_code");
  });
}
if (params.get("host_token")) {
  state.hostToken = params.get("host_token");
  localStorage.setItem("pokerArenaHostToken", state.hostToken);
  document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("hostTokenInput").value = state.hostToken;
  });
}

function connect() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const tokenQuery = state.seatToken ? `?seat_token=${encodeURIComponent(state.seatToken)}` : "";
  const socket = new WebSocket(`${protocol}://${window.location.host}/ws${tokenQuery}`);
  state.socket = socket;
  socket.addEventListener("open", () => setConnection("online"));
  socket.addEventListener("close", () => {
    setConnection("offline");
    setTimeout(connect, 1200);
  });
  socket.addEventListener("message", (event) => {
    if (event.data === "pong") return;
    state.snapshot = JSON.parse(event.data);
    render();
  });
}

function setConnection(text) {
  document.getElementById("connectionStatus").textContent = text;
}

function cardEl(text) {
  const card = document.createElement("span");
  card.className = `card ${/[hd]$/i.test(text) ? "red" : ""}`;
  card.textContent = text;
  return card;
}

function render() {
  const snap = state.snapshot;
  if (!snap) return;
  document.getElementById("roomTitle").textContent = `Room ${snap.room_code}`;
  document.getElementById("gameStatus").textContent = snap.paused_reason || snap.status;
  document.getElementById("blindStatus").textContent = `${snap.settings.small_blind} / ${snap.settings.big_blind}`;
  document.getElementById("roomCodeInput").value ||= snap.room_code;

  renderSeatOptions();
  renderSeats();
  renderBoard();
  renderActions();
  renderLog();
}

function renderSeatOptions() {
  const snap = state.snapshot;
  const humanSelect = document.getElementById("seatSelect");
  const botSelect = document.getElementById("botSeatSelect");
  humanSelect.innerHTML = "";
  botSelect.innerHTML = "";
  snap.seats.forEach((seat) => {
    if (seat.kind !== "empty") return;
    const label = `Seat ${seat.seat_id + 1}`;
    humanSelect.add(new Option(label, seat.seat_id));
    botSelect.add(new Option(label, seat.seat_id));
  });
}

function renderSeats() {
  const grid = document.getElementById("seat-grid");
  const template = document.getElementById("seatTemplate");
  grid.innerHTML = "";
  state.snapshot.seats.forEach((seat) => {
    const node = template.content.firstElementChild.cloneNode(true);
    node.dataset.seat = seat.seat_id;
    node.classList.add(seat.kind);
    if (state.snapshot.current_actor === seat.seat_id) node.classList.add("current");
    if (seat.kind === "human" && state.seatToken && state.snapshot.private_hole_cards && seat.in_hand) {
      const actorOwnsPrivateCards = state.snapshot.legal_actions || state.snapshot.current_actor !== seat.seat_id;
      if (actorOwnsPrivateCards) node.classList.add("mine");
    }
    node.querySelector(".seat-number").textContent = `Seat ${seat.seat_id + 1}`;
    node.querySelector(".seat-name").textContent = seat.nickname || "Open";
    node.querySelector(".seat-stack").textContent = `${seat.stack} chips`;
    const badges = [];
    if (state.snapshot.button === seat.seat_id) badges.push("BTN");
    if (state.snapshot.small_blind === seat.seat_id) badges.push("SB");
    if (state.snapshot.big_blind === seat.seat_id) badges.push("BB");
    if (seat.folded) badges.push("folded");
    if (seat.all_in) badges.push("all-in");
    node.querySelector(".seat-badges").textContent = badges.join(" ");
    grid.appendChild(node);
  });
}

function renderBoard() {
  const board = document.getElementById("boardCards");
  board.innerHTML = "";
  state.snapshot.board.forEach((text) => board.appendChild(cardEl(text)));
  document.getElementById("potDisplay").textContent = `Pot ${state.snapshot.pot}`;
  document.getElementById("streetDisplay").textContent = state.snapshot.street;

  const privateCards = document.getElementById("privateCards");
  privateCards.innerHTML = "";
  const cards = state.snapshot.private_hole_cards || [];
  cards.forEach((text) => privateCards.appendChild(cardEl(text)));
}

function renderActions() {
  const legal = state.snapshot.legal_actions;
  const fold = document.getElementById("foldButton");
  const check = document.getElementById("checkButton");
  const call = document.getElementById("callButton");
  const raise = document.getElementById("raiseButton");
  const raiseByInput = document.getElementById("raiseByInput");
  [fold, check, call, raise, raiseByInput, ...document.querySelectorAll("[data-raise]")].forEach((el) => {
    el.disabled = !legal;
  });
  if (!legal) return;

  fold.disabled = !legal.can_fold;
  check.disabled = !legal.can_check;
  call.disabled = !legal.can_call;
  call.textContent = legal.can_call ? `Call ${legal.call_amount}` : "Call";
  raise.disabled = !legal.can_raise;
  const minRaiseBy = legal.can_raise ? legal.min_raise_to - legal.current_bet : 1;
  const maxRaiseBy = legal.can_raise ? legal.max_raise_to - legal.current_bet : 1;
  raiseByInput.min = minRaiseBy;
  raiseByInput.max = maxRaiseBy;
  raiseByInput.placeholder = legal.can_raise ? `${minRaiseBy}-${maxRaiseBy}` : "";
}

function renderLog() {
  const log = document.getElementById("handLog");
  log.innerHTML = "";
  let currentBet = 0;
  state.snapshot.log.slice(-24).forEach((event) => {
    const item = document.createElement("li");
    const formatted = formatEvent(event, currentBet);
    currentBet = formatted.currentBet;
    item.textContent = formatted.text;
    log.appendChild(item);
  });
}

function formatEvent(event, currentBet) {
  const data = event.data || {};
  if (event.event_type === "small_blind" || event.event_type === "big_blind") {
    return { text: `Seat ${data.seat_id + 1}: ${event.event_type.replace("_", " ")} ${data.amount}`, currentBet: Math.max(currentBet, data.amount || 0) };
  }
  if (event.event_type === "action") {
    if (data.action.type === "raise_to") {
      const raiseBy = Math.max(0, Number(data.action.total) - currentBet);
      return { text: `Seat ${data.seat_id + 1}: raise_by ${raiseBy} (to ${data.action.total})`, currentBet: Number(data.action.total) };
    }
    return { text: `Seat ${data.seat_id + 1}: ${data.action.type}`, currentBet };
  }
  if (event.event_type === "street_dealt") return { text: `${data.street}: ${(data.board || []).join(" ")}`, currentBet: 0 };
  if (event.event_type === "pot_awarded") return { text: `Seat ${data.seat_id + 1} won ${data.amount}`, currentBet };
  if (event.event_type === "hand_started") return { text: `Hand ${data.hand_number}`, currentBet: 0 };
  return { text: event.event_type.replaceAll("_", " "), currentBet };
}

async function postJSON(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}

async function joinSeat() {
  const payload = await postJSON("/api/join", {
    room_code: document.getElementById("roomCodeInput").value.trim(),
    nickname: document.getElementById("nicknameInput").value.trim(),
    seat_id: Number(document.getElementById("seatSelect").value),
  });
  state.seatToken = payload.seat_token;
  localStorage.setItem("pokerArenaSeatToken", state.seatToken);
  state.socket.close();
}

async function addBot() {
  state.hostToken = document.getElementById("hostTokenInput").value.trim();
  localStorage.setItem("pokerArenaHostToken", state.hostToken);
  await postJSON("/api/host/bots", {
    host_token: state.hostToken,
    seat_id: Number(document.getElementById("botSeatSelect").value),
  });
}

async function submitAction(action) {
  if (!state.seatToken) return;
  await postJSON("/api/action", {
    seat_token: state.seatToken,
    action,
  });
}

function setQuickRaise(kind) {
  const legal = state.snapshot && state.snapshot.legal_actions;
  if (!legal || !legal.can_raise) return;
  let value = legal.min_raise_to;
  if (kind === "half") value = legal.current_bet + Math.round(state.snapshot.pot / 2);
  if (kind === "pot") value = legal.current_bet + state.snapshot.pot;
  if (kind === "allin") value = legal.max_raise_to;
  value = Math.max(legal.min_raise_to, Math.min(legal.max_raise_to, value));
  document.getElementById("raiseByInput").value = Math.max(1, value - legal.current_bet);
}

function raiseByToRaiseTo(raiseBy) {
  const legal = state.snapshot && state.snapshot.legal_actions;
  if (!legal || !legal.can_raise) return null;
  const total = legal.current_bet + Number(raiseBy);
  return Math.max(legal.min_raise_to, Math.min(legal.max_raise_to, total));
}

function downloadLog() {
  const hostToken = document.getElementById("hostTokenInput").value.trim() || state.hostToken;
  if (!hostToken) return;
  window.location.href = `/api/logs/session.json?host_token=${encodeURIComponent(hostToken)}`;
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("joinButton").addEventListener("click", () => joinSeat().catch(alert));
  document.getElementById("addBotButton").addEventListener("click", () => addBot().catch(alert));
  document.getElementById("downloadLogButton").addEventListener("click", downloadLog);
  document.getElementById("foldButton").addEventListener("click", () => submitAction({ type: "fold", total: null }).catch(alert));
  document.getElementById("checkButton").addEventListener("click", () => submitAction({ type: "check", total: null }).catch(alert));
  document.getElementById("callButton").addEventListener("click", () => submitAction({ type: "call", total: null }).catch(alert));
  document.getElementById("raiseButton").addEventListener("click", () => {
    const total = raiseByToRaiseTo(Number(document.getElementById("raiseByInput").value));
    submitAction({ type: "raise_to", total }).catch(alert);
  });
  document.querySelectorAll("[data-raise]").forEach((button) => {
    button.addEventListener("click", () => setQuickRaise(button.dataset.raise));
  });
  if (state.hostToken) document.getElementById("hostTokenInput").value = state.hostToken;
  connect();
});
