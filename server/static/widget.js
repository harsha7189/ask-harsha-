/* Ask Harsha chat pop-up. Add to any website with one line:
 *
 *   <script src="https://ask-harsha.onrender.com/widget.js" defer></script>
 *
 * Optional attributes on that tag:
 *   data-label="Ask my AI"                       text on the button (desktop)
 *   data-greeting="Hi! Ask me about my work."    the small bubble shown once per visit ("" to turn it off)
 *   data-side="left"                             put it bottom-left instead of bottom-right
 *
 * It adds a floating button; tapping it opens the agent in a panel (full screen on phones). The chat itself
 * loads only when someone opens the panel, and the page quietly wakes the server on load so the first answer
 * isn't slowed by a cold start. Everything lives in a shadow root, so the host page's styles can't break it.
 */
(function () {
  'use strict';
  if (window.__askHarshaWidget) return;
  window.__askHarshaWidget = true;

  var me = document.currentScript || (function () { var s = document.querySelectorAll('script[src*="widget.js"]'); return s[s.length - 1]; })();
  var origin = new URL(me && me.src ? me.src : location.href).origin;
  var label = (me && me.getAttribute('data-label')) || 'Ask my AI';
  var greetAttr = me && me.getAttribute('data-greeting');
  var greeting = greetAttr === null ? "Hi! I'm Harsha's AI agent. Ask me anything about his work." : greetAttr;
  var left = me && me.getAttribute('data-side') === 'left';

  // Wake the server now (free hosting sleeps when idle), so the chat is ready by the time someone opens it.
  try { fetch(origin + '/healthz', { mode: 'no-cors', cache: 'no-store' }).catch(function () {}); } catch (e) {}

  var host = document.createElement('div');
  host.id = 'ask-harsha-widget';
  host.style.cssText = 'position:fixed;z-index:2147483000;bottom:0;' + (left ? 'left:0' : 'right:0') + ';width:0;height:0';
  var root = host.attachShadow ? host.attachShadow({ mode: 'open' }) : host;
  var side = left ? 'left' : 'right';

  root.innerHTML =
    '<style>' +
    ':host{all:initial}' +
    '*{box-sizing:border-box}' +
    '.wrap{--red:#e50914;--red-hot:#ff2a36;--bg:#0b0b0b;--panel:#171717;--line:#2e2e2e;--fg:#f5f5f1;--muted:#a3a3a3;font-family:"Instrument Sans",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:var(--fg)}' +
    '.fab{position:fixed;' + side + ':max(18px,env(safe-area-inset-' + side + ',0px));bottom:max(18px,env(safe-area-inset-bottom,0px));display:flex;align-items:center;gap:10px;height:56px;padding:0 20px 0 8px;border:0;border-radius:999px;background:var(--red);color:#fff;font-weight:700;font-size:15px;line-height:1;font-family:inherit;cursor:pointer;box-shadow:0 10px 30px rgba(0,0,0,.45),0 0 0 1px rgba(255,255,255,.08) inset;transition:transform .15s,background-color .15s}' +
    '.fab:hover{background:var(--red-hot);transform:translateY(-2px)}' +
    '.fab:focus-visible{outline:3px solid #fff;outline-offset:3px}' +
    '.fab .m{width:40px;height:40px;border-radius:50%;background:#fff;color:var(--red);display:grid;place-items:center;font:900 22px/1 "Big Shoulders Display","Arial Narrow",Impact,sans-serif}' +
    '.fab .dot{position:absolute;top:6px;' + (left ? 'left' : 'right') + ':8px;width:10px;height:10px;border-radius:50%;background:#46d369;box-shadow:0 0 0 2px var(--red)}' +
    '@media (max-width:520px){.fab{padding:0 8px;width:56px;justify-content:center}.fab .t{display:none}}' +
    '.bubble{position:fixed;' + side + ':max(18px,env(safe-area-inset-' + side + ',0px));bottom:calc(max(18px,env(safe-area-inset-bottom,0px)) + 68px);max-width:min(280px,calc(100vw - 36px));background:#fff;color:#111;border-radius:14px;padding:12px 34px 12px 14px;font-weight:500;font-size:14px;line-height:1.4;font-family:inherit;box-shadow:0 12px 30px rgba(0,0,0,.35);cursor:pointer;animation:pop .3s ease}' +
    '.bubble button{position:absolute;top:4px;right:4px;width:28px;height:28px;border:0;background:none;color:#666;font-size:18px;cursor:pointer;border-radius:50%}' +
    '.bubble button:hover{background:#eee}' +
    '.panel{position:fixed;' + side + ':max(18px,env(safe-area-inset-' + side + ',0px));bottom:calc(max(18px,env(safe-area-inset-bottom,0px)) + 68px);width:400px;height:min(640px,calc(100vh - 110px));background:var(--bg);border:1px solid var(--line);border-radius:18px;overflow:hidden;display:flex;flex-direction:column;box-shadow:0 24px 60px rgba(0,0,0,.55);animation:pop .22s ease}' +
    '@media (max-width:520px){.panel{inset:0;width:auto;height:auto;border-radius:0;border:0}.wrap.open .fab{display:none}}' +
    '.bar{display:flex;align-items:center;gap:10px;padding:10px 10px 10px 14px;padding-top:max(10px,env(safe-area-inset-top,0px));border-bottom:1px solid var(--line);background:var(--bg)}' +
    '.bar b{font-size:15px;flex:1}' +
    '.bar b small{display:block;font-weight:500;font-size:12px;color:var(--muted)}' +
    '.bar a,.bar button{display:grid;place-items:center;width:38px;height:38px;border-radius:10px;border:0;background:none;color:var(--muted);cursor:pointer;text-decoration:none;font-size:18px}' +
    '.bar a:hover,.bar button:hover{color:var(--fg);background:#232323}' +
    '.bar a:focus-visible,.bar button:focus-visible{outline:2px solid #fff}' +
    'iframe{flex:1;width:100%;border:0;background:var(--bg)}' +
    '@keyframes pop{from{opacity:0;transform:translateY(8px) scale(.98)}to{opacity:1;transform:none}}' +
    '@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}' +
    '[hidden]{display:none!important}' +
    '</style>' +
    '<div class="wrap">' +
    '<div class="bubble" id="bubble" role="status" hidden><span id="btext"></span><button type="button" id="bclose" aria-label="Dismiss">×</button></div>' +
    '<div class="panel" id="panel" role="dialog" aria-label="Chat with Harsha\'s AI agent" hidden>' +
    '<div class="bar"><b>Ask Harsha<small>AI agent · answers from his verified record</small></b>' +
    '<a id="full" href="' + origin + '/" target="_blank" rel="noopener" aria-label="Open the full app in a new tab" title="Open full app">↗</a>' +
    '<button type="button" id="close" aria-label="Close chat" title="Close">×</button></div>' +
    '</div>' +
    '<button class="fab" id="fab" type="button" aria-expanded="false" aria-controls="panel"><span class="m">H</span><span class="t"></span><span class="dot" aria-hidden="true"></span></button>' +
    '</div>';

  var $ = function (id) { return root.getElementById ? root.getElementById(id) : root.querySelector('#' + id); };
  var fab = $('fab'), panel = $('panel'), bubble = $('bubble'), wrap = root.querySelector('.wrap');
  fab.querySelector('.t').textContent = label;
  fab.setAttribute('aria-label', label + ': chat with Harsha\'s AI agent');
  var frame = null;

  function open(question) {
    if (!frame) {
      frame = document.createElement('iframe');
      frame.title = 'Ask Harsha chat';
      frame.allow = 'microphone; clipboard-write';
      frame.src = origin + '/?embed=1' + (question ? '&q=' + encodeURIComponent(question) : '');
      panel.appendChild(frame);
    }
    panel.hidden = false; bubble.hidden = true; fab.setAttribute('aria-expanded', 'true'); wrap.classList.add('open');
    try { sessionStorage.setItem('askh_widget_seen', '1'); } catch (e) {}
    setTimeout(function () { try { frame.focus(); } catch (e) {} }, 50);
  }
  function close() { panel.hidden = true; wrap.classList.remove('open'); fab.setAttribute('aria-expanded', 'false'); fab.focus(); }

  fab.addEventListener('click', function () { panel.hidden ? open() : close(); });
  $('close').addEventListener('click', close);
  bubble.addEventListener('click', function (e) { if (e.target.id !== 'bclose') open(); });
  $('bclose').addEventListener('click', function (e) { e.stopPropagation(); bubble.hidden = true; try { sessionStorage.setItem('askh_widget_seen', '1'); } catch (x) {} });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && !panel.hidden) close(); });
  window.addEventListener('message', function (e) {
    if (e.origin !== origin || !e.data || e.data.type !== 'askh:close') return;
    close();
  });

  // Let the host page open the chat with a question: window.AskHarsha.open("What did he ship?")
  window.AskHarsha = { open: open, close: close };

  function mount() {
    document.body.appendChild(host);
    var seen = false;
    try { seen = sessionStorage.getItem('askh_widget_seen') === '1'; } catch (e) {}
    if (greeting && !seen) {
      $('btext').textContent = greeting;
      setTimeout(function () { if (panel.hidden) bubble.hidden = false; }, 3500);
    }
  }
  if (document.body) mount(); else document.addEventListener('DOMContentLoaded', mount);
})();
