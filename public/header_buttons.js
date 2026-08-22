/**
 * Injects "Mock Accounts" and "Logs" buttons into the Chainlit header,
 * next to the existing "Readme" button.
 *
 * On click each button submits a hidden slash-command to the Chainlit chat
 * input, which the Python on_message handler intercepts before it reaches
 * the LLM.
 */

(function () {
  const COMMANDS = {
    "Mock Accounts": "/__mock_accounts",
    "Logs": "/__logs",
  };

  const BTN_STYLE = `
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 12px;
    border-radius: 6px;
    border: 1px solid rgba(255,255,255,0.15);
    background: transparent;
    color: inherit;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    white-space: nowrap;
    transition: background 0.15s;
  `;

  const ICONS = {
    "Mock Accounts": "🔑",
    "Logs": "🔍",
  };

  function isAuthenticated() {
    // During auth flow, Chainlit replaces the normal composer with AskUserMessage /
    // AskActionMessage elements. After auth completes the standard message composer
    // (textarea inside a <form>) is the only input in the page.
    // We use this as a lightweight proxy for "auth complete" so that header
    // buttons cannot submit commands while auth steps are still in progress.
    const form = document.querySelector('form');
    if (!form) return false;
    const textarea = form.querySelector('textarea[placeholder]');
    return !!textarea;
  }

  function submitCommand(cmd) {
    if (!isAuthenticated()) {
      // Auth flow still active; ignore click
      return;
    }
    // React's synthetic event system requires using the native setter
    const textarea = document.querySelector('textarea[placeholder]');
    if (!textarea) return;

    const nativeSetter = Object.getOwnPropertyDescriptor(
      window.HTMLTextAreaElement.prototype, 'value'
    ).set;
    nativeSetter.call(textarea, cmd);
    textarea.dispatchEvent(new Event('input', { bubbles: true }));
    textarea.dispatchEvent(new Event('change', { bubbles: true }));

    // Give React one tick to process, then click the send button
    setTimeout(() => {
      const sendBtn = textarea.closest('form')?.querySelector('button[type="submit"]')
        || document.querySelector('button[data-testid="send-button"]')
        || [...document.querySelectorAll('button')].find(b =>
            b.querySelector('svg') && b.closest('[class*="composer"], [class*="input"]')
          );
      if (sendBtn) {
        sendBtn.click();
      } else {
        // fallback: dispatch Enter key
        textarea.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', keyCode: 13, bubbles: true }));
        textarea.dispatchEvent(new KeyboardEvent('keyup',  { key: 'Enter', keyCode: 13, bubbles: true }));
      }
    }, 50);
  }

  function createButton(label) {
    const btn = document.createElement('button');
    btn.textContent = ICONS[label] + ' ' + label;
    btn.setAttribute('style', BTN_STYLE);
    btn.setAttribute('title', label);
    btn.addEventListener('mouseenter', () => {
      btn.style.background = 'rgba(255,255,255,0.08)';
    });
    btn.addEventListener('mouseleave', () => {
      btn.style.background = 'transparent';
    });
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      if (!isAuthenticated()) {
        btn.style.opacity = '0.4';
        btn.title = 'Complete authentication first';
        return;
      }
      submitCommand(COMMANDS[label]);
    });
    return btn;
  }

  function injectButtons() {
    // Avoid duplicate injection
    if (document.getElementById('pp-header-btn-logs')) return;

    // Find the Readme button — it's typically a <button> or <a> in the header
    const readmeEl = [...document.querySelectorAll('button, a')]
      .find(el => el.textContent.trim() === 'Readme');
    if (!readmeEl) return;

    const container = readmeEl.parentElement;
    if (!container) return;

    Object.keys(COMMANDS).reverse().forEach(label => {
      const btn = createButton(label);
      btn.id = label === 'Logs' ? 'pp-header-btn-logs' : 'pp-header-btn-mock';
      // Insert before the Readme button so the order is: Mock Accounts | Logs | Readme
      container.insertBefore(btn, readmeEl);

      // Small gap between our buttons and Readme
      const spacer = document.createElement('span');
      spacer.style.cssText = 'display:inline-block;width:6px;';
      container.insertBefore(spacer, readmeEl);
    });
  }

  // Watch for the header to appear (Chainlit is a SPA so it renders async)
  const observer = new MutationObserver(() => {
    injectButtons();
  });
  observer.observe(document.body, { childList: true, subtree: true });

  // Also try immediately in case the header already exists
  if (document.readyState !== 'loading') {
    injectButtons();
  } else {
    document.addEventListener('DOMContentLoaded', injectButtons);
  }
})();
