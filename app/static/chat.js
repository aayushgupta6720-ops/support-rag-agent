// Tidewell help center: the chat widget and the page controls that open it.
// Talks to POST /chat on the same origin. Every message is built from text
// nodes, never innerHTML, so nothing in an answer can inject markup.
(function () {
  "use strict";

  const STORAGE_KEY = "tidewell-chat";
  const MAX_CHARS = 2000;
  // The server finishes a /chat within ~225s even when Gemini retries; past
  // that, something is wrong and the visitor shouldn't keep waiting.
  const REQUEST_TIMEOUT_MS = 240000;

  // Doc ids the API returns as sources, and the article titles to show.
  const SOURCE_TITLES = {
    "password-reset": "Resetting your password",
    "two-factor-auth": "Two-factor authentication",
    "sign-in-and-lockouts": "Sign-in problems",
    "account-deletion": "Deleting your account",
    "billing-refunds": "Billing and refunds",
    "invoices-and-receipts": "Invoices and receipts",
    "team-members": "Team members and roles",
    "email-notifications": "Email notifications",
    "api-keys": "Managing API keys",
    "api-rate-limits": "API rate limits",
    "data-export": "Exporting your data",
  };

  const SUGGESTIONS = [
    "How do I reset my password?",
    "Can I get a refund on my annual plan?",
    "What are the API rate limits?",
    "How do I invite someone to my team?",
  ];

  const $ = (id) => document.getElementById(id);
  const panel = $("chat-panel");
  const launcher = $("chat-launcher");
  const log = $("chat-log");
  const form = $("chat-composer");
  const input = $("chat-input");
  const sendButton = $("chat-send");
  const counter = $("chat-count");
  const greeting = $("chat-greeting");
  const articleView = $("chat-article");
  const articleBody = $("chat-article-body");

  // ---- state, kept for the tab's lifetime so a refresh keeps the chat ------

  let state = { sessionId: null, messages: [], greetingDismissed: false };
  let busy = false;

  function load() {
    try {
      const saved = JSON.parse(sessionStorage.getItem(STORAGE_KEY) || "null");
      if (saved && Array.isArray(saved.messages)) state = Object.assign(state, saved);
    } catch (e) { /* storage blocked or corrupt: start fresh */ }
  }

  function save() {
    try { sessionStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch (e) { /* not essential */ }
  }

  // ---- rendering --------------------------------------------------------------

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  // **bold** and `code` inside a line, as text nodes and elements.
  function appendInline(parent, text) {
    const pattern = /(\*\*[^*]+\*\*|`[^`]+`)/g;
    let last = 0;
    let match;
    while ((match = pattern.exec(text)) !== null) {
      if (match.index > last) parent.appendChild(document.createTextNode(text.slice(last, match.index)));
      const token = match[0];
      parent.appendChild(token.startsWith("**") ? el("strong", null, token.slice(2, -2)) : el("code", null, token.slice(1, -1)));
      last = match.index + token.length;
    }
    if (last < text.length) parent.appendChild(document.createTextNode(text.slice(last)));
  }

  // Paragraphs, bulleted and numbered lists: the formatting answers use.
  // softWrap joins a paragraph's lines with spaces, for the help articles,
  // whose source text is hard-wrapped; in answers a newline is a break.
  function formatAnswer(text, softWrap) {
    const fragment = document.createDocumentFragment();
    let list = null;
    let paragraph = null;
    for (const raw of text.replace(/\r/g, "").split("\n")) {
      const line = raw.trim();
      const bullet = line.match(/^[-*•]\s+(.*)$/);
      const numbered = line.match(/^\d+[.)]\s+(.*)$/);
      if (!line) { list = null; paragraph = null; continue; }
      if (bullet || numbered) {
        const kind = bullet ? "ul" : "ol";
        if (!list || list.tagName.toLowerCase() !== kind) { list = el(kind); fragment.appendChild(list); }
        const item = el("li");
        appendInline(item, (bullet || numbered)[1]);
        list.appendChild(item);
        paragraph = null;
        continue;
      }
      list = null;
      if (paragraph) paragraph.appendChild(softWrap ? document.createTextNode(" ") : document.createElement("br"));
      else { paragraph = el("p"); fragment.appendChild(paragraph); }
      appendInline(paragraph, line);
    }
    return fragment;
  }

  function renderMessage(message) {
    const wrap = el("div", "msg msg-" + message.role);
    const bubble = el("div", "bubble");
    if (message.role === "bot") bubble.appendChild(formatAnswer(message.text));
    else bubble.textContent = message.text;
    wrap.appendChild(bubble);
    if (message.sources && message.sources.length) {
      const sources = el("div", "sources", "From:");
      for (const id of message.sources) {
        const chip = el("button", "source-chip", SOURCE_TITLES[id] || id);
        chip.type = "button";
        chip.setAttribute("aria-label", "Read the article: " + (SOURCE_TITLES[id] || id));
        chip.addEventListener("click", () => openArticle(id));
        sources.appendChild(chip);
      }
      wrap.appendChild(sources);
    }
    if (message.role === "bot" && message.answerId) wrap.appendChild(renderRating(message));
    log.appendChild(wrap);
  }

  const THUMB = "M7 11v9M7 11l4-7c1.3 0 2 1 2 2.2V10h5.3a2 2 0 0 1 2 2.3l-1.2 6A2 2 0 0 1 17 20H7";

  function thumbIcon(down) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("aria-hidden", "true");
    if (down) svg.setAttribute("class", "thumb-down");
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", THUMB);
    svg.appendChild(path);
    return svg;
  }

  // Rating sends the rated question and answer, so the note says so.
  function renderRating(message) {
    const row = el("div", "rating");
    if (message.rated) {
      row.appendChild(el("span", "rating-done", "Thanks for the feedback."));
      return row;
    }
    row.appendChild(el("span", "rating-label", "Helpful?"));
    for (const rating of ["up", "down"]) {
      const button = el("button", "rating-button");
      button.type = "button";
      button.setAttribute("aria-label", rating === "up" ? "Yes, this helped" : "No, this didn't help");
      button.title = "Rating saves this question and answer so we can improve the assistant.";
      button.appendChild(thumbIcon(rating === "down"));
      button.addEventListener("click", () => rate(message, rating, row));
      row.appendChild(button);
    }
    return row;
  }

  async function rate(message, rating, row) {
    row.querySelectorAll("button").forEach((button) => { button.disabled = true; });
    try {
      const response = await fetch("/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          answer_id: message.answerId, rating: rating,
          question: message.question, answer: message.text, sources: message.sources || [],
        }),
      });
      if (!response.ok) throw new Error("HTTP " + response.status);
      message.rated = rating;
      save();
      row.replaceWith(renderRating(message));
    } catch (error) {
      row.querySelectorAll("button").forEach((button) => { button.disabled = false; });
      const label = row.querySelector(".rating-label");
      if (label) label.textContent = "Couldn't send that. Try again?";
    }
  }

  // ---- articles -----------------------------------------------------------------------

  async function openArticle(docId) {
    articleBody.replaceChildren(el("p", "article-loading", "Loading the article…"));
    panel.classList.add("showing-article");
    articleView.hidden = false;
    $("chat-article-back").focus();
    try {
      const response = await fetch("/articles/" + encodeURIComponent(docId));
      if (!response.ok) throw new Error("HTTP " + response.status);
      const article = await response.json();
      articleBody.replaceChildren(el("h3", "article-title", article.title), formatAnswer(article.body, true));
    } catch (error) {
      articleBody.replaceChildren(el("p", "article-loading", "Couldn't load that article. Please try again."));
    }
  }

  function closeArticle() {
    if (articleView.hidden) return false;
    articleView.hidden = true;
    panel.classList.remove("showing-article");
    input.focus();
    return true;
  }

  function renderWelcome() {
    const wrap = el("div", "msg msg-bot");
    const bubble = el("div", "bubble");
    bubble.appendChild(formatAnswer(
      "Hi! I'm Tidewell's support assistant. I answer from our help articles, " +
      "and I'll tell you when they don't cover something. What can I help with?"
    ));
    wrap.appendChild(bubble);
    const chips = el("div", "suggestions");
    for (const question of SUGGESTIONS) {
      const chip = el("button", "suggestion", question);
      chip.type = "button";
      chip.addEventListener("click", () => send(question));
      chips.appendChild(chip);
    }
    wrap.appendChild(chips);
    log.appendChild(wrap);
  }

  function renderAll() {
    log.replaceChildren();
    renderWelcome();
    state.messages.forEach(renderMessage);
    scrollToEnd();
  }

  function scrollToEnd() {
    log.scrollTop = log.scrollHeight;
  }

  function addMessage(message) {
    state.messages.push(message);
    save();
    renderMessage(message);
    scrollToEnd();
  }

  // ---- talking to the API -------------------------------------------------------

  function showTyping() {
    const wrap = el("div", "msg msg-bot");
    wrap.setAttribute("aria-label", "The assistant is typing");
    const bubble = el("div", "bubble typing");
    bubble.append(el("span"), el("span"), el("span"));
    const note = el("div", "typing-note");
    wrap.append(bubble, note);
    log.appendChild(wrap);
    scrollToEnd();
    const timers = [
      setTimeout(() => { note.textContent = "Still working on it…"; scrollToEnd(); }, 10000),
      setTimeout(() => {
        note.textContent = "This is taking longer than usual: the demo server may be waking up, or the AI model may be busy.";
        scrollToEnd();
      }, 30000),
    ];
    return () => { timers.forEach(clearTimeout); wrap.remove(); };
  }

  // The API's own `detail` is written for people (rate limits, quota, timeouts).
  async function errorText(response) {
    let detail = null;
    try { detail = (await response.json()).detail; } catch (e) { /* not JSON */ }
    if (typeof detail === "string") return detail;
    if (response.status === 422) return "Please type a question of up to 2,000 characters.";
    return "Something went wrong on our side (HTTP " + response.status + "). Please try again in a moment.";
  }

  async function send(text) {
    const query = text.trim();
    if (!query || busy) return;
    openPanel();
    closeArticle();
    busy = true;
    updateComposer();
    addMessage({ role: "user", text: query });
    const hideTyping = showTyping();
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query, session_id: state.sessionId }),
        signal: controller.signal,
      });
      hideTyping();
      if (response.ok) {
        const body = await response.json();
        state.sessionId = body.session_id || state.sessionId;
        addMessage({
          role: "bot", text: body.answer, sources: body.sources || [],
          answerId: body.answer_id, question: query,
        });
      } else {
        addMessage({ role: "system", text: await errorText(response) });
      }
    } catch (error) {
      hideTyping();
      addMessage({
        role: "system",
        text: error.name === "AbortError"
          ? "The assistant took too long to answer. Please try again."
          : "Couldn't reach the support assistant. Check your connection and try again.",
      });
    } finally {
      clearTimeout(timeout);
      busy = false;
      updateComposer();
      input.focus();
    }
  }

  // ---- composer -------------------------------------------------------------------

  function updateComposer() {
    const length = input.value.length;
    sendButton.disabled = busy || !input.value.trim();
    input.setAttribute("aria-busy", busy ? "true" : "false");
    counter.textContent = length > MAX_CHARS - 200 ? (MAX_CHARS - length) + " characters left." : "";
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 140) + "px";
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = input.value;
    if (!text.trim() || busy) return;
    input.value = "";
    updateComposer();
    send(text);
  });

  input.addEventListener("input", updateComposer);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  // ---- opening and closing --------------------------------------------------------

  function openPanel() {
    if (!panel.hidden) return;
    panel.hidden = false;
    launcher.setAttribute("aria-expanded", "true");
    dismissGreeting();
    scrollToEnd();
    input.focus();
  }

  function closePanel() {
    panel.hidden = true;
    launcher.setAttribute("aria-expanded", "false");
    launcher.focus();
  }

  function dismissGreeting() {
    greeting.hidden = true;
    if (!state.greetingDismissed) { state.greetingDismissed = true; save(); }
  }

  launcher.addEventListener("click", () => (panel.hidden ? openPanel() : closePanel()));
  $("chat-close").addEventListener("click", closePanel);
  $("chat-new").addEventListener("click", () => {
    if (busy) return;
    closeArticle();
    state.sessionId = null;
    state.messages = [];
    save();
    renderAll();
    input.focus();
  });
  greeting.querySelector(".chat-greeting-close").addEventListener("click", dismissGreeting);
  $("chat-article-back").addEventListener("click", closeArticle);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !panel.hidden && !closeArticle()) closePanel();
  });

  // Page controls: the hero search box, "popular" chips, topic cards, and
  // anything marked data-open-chat.
  $("hero-ask").addEventListener("submit", (event) => {
    event.preventDefault();
    const box = $("hero-input");
    const text = box.value;
    box.value = "";
    send(text);
  });
  document.querySelectorAll("[data-ask]").forEach((button) => {
    button.addEventListener("click", () => send(button.getAttribute("data-ask")));
  });
  document.querySelectorAll("[data-open-chat]").forEach((button) => {
    button.addEventListener("click", openPanel);
  });

  // ---- start ------------------------------------------------------------------------

  load();
  renderAll();
  updateComposer();
  if (!state.greetingDismissed && !state.messages.length) {
    setTimeout(() => { if (panel.hidden) greeting.hidden = false; }, 4000);
  }
})();
