/**
 * AI CCTV — Theme Management System
 * Supports Dark (Cyber Command Center) and Light (Crisp Tech Slate) themes
 * with instant FOUC-prevention, persistence, and reactive UI updates.
 */

(function () {
    // Immediate execution before DOM render to prevent FOUC
    try {
        const savedTheme = localStorage.getItem('cctv_theme') || 'dark';
        document.documentElement.setAttribute('data-theme', savedTheme);
    } catch (e) {
        document.documentElement.setAttribute('data-theme', 'dark');
    }
})();

function getAppTheme() {
    return document.documentElement.getAttribute('data-theme') || localStorage.getItem('cctv_theme') || 'dark';
}

function setAppTheme(theme) {
    const validTheme = theme === 'light' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', validTheme);
    try {
        localStorage.setItem('cctv_theme', validTheme);
    } catch (e) {
        console.warn('localStorage access failed:', e);
    }
    updateThemeToggleUI();
}

function toggleAppTheme() {
    const current = getAppTheme();
    const next = current === 'dark' ? 'light' : 'dark';
    setAppTheme(next);
}

function updateThemeToggleUI() {
    const current = getAppTheme();
    const isLight = current === 'light';
    const btns = document.querySelectorAll('.theme-toggle-btn');
    btns.forEach(btn => {
        btn.setAttribute('aria-checked', isLight ? 'true' : 'false');
        btn.setAttribute('title', isLight ? 'สลับเป็นธีมมืด (Switch to Dark Theme)' : 'สลับเป็นธีมสว่าง (Switch to Light Theme)');
        btn.setAttribute('aria-label', isLight ? 'สลับเป็นธีมมืด' : 'สลับเป็นธีมสว่าง');
    });
}

// Sync toggle state once DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', updateThemeToggleUI);
} else {
    updateThemeToggleUI();
}
