import React, { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';

/* ───────────────────────────────────────────
   CONFIGURATION — Edit these to personalise
   ─────────────────────────────────────────── */

const PROFILE = {
  name: 'Jose Garcia',
  title: 'Statistician',
  tagline: 'Building things that matter.',
};

const ABOUT_CARDS = [
  { label: 'Location', value: '📍 San Francisco, CA' },
  { label: 'Focus', value: '🎯 Full-Stack Development' },
  { label: 'Experience', value: '💼 3+ Years' },
  { label: 'Education', value: '🎓 B.S. Computer Science' },
];

const SKILLS = ['React', 'JavaScript', 'TypeScript', 'Node.js', 'Python', 'CSS', 'Git', 'AWS'];

const PROJECTS = [
  {
    title: 'E-Commerce Platform',
    description: 'A full-stack e-commerce solution with cart, checkout, and payment integration.',
    tech: ['React', 'Node.js', 'MongoDB', 'Stripe'],
  },
  {
    title: 'Weather Dashboard',
    description: 'Real-time weather dashboard with interactive maps and 7-day forecasts.',
    tech: ['React', 'OpenWeather API', 'Chart.js'],
  },
  {
    title: 'Task Manager',
    description: 'Collaborative task management app with real-time updates and team features.',
    tech: ['Next.js', 'PostgreSQL', 'WebSockets'],
  },
  {
    title: 'Portfolio OS',
    description: 'This very portfolio — a macOS-inspired interactive OS experience.',
    tech: ['React', 'CSS3', 'JavaScript'],
  },
];

const CONTACT_LINKS = [
  { icon: '✉️', label: 'Email', value: 'hello@example.com', href: 'mailto:hello@example.com' },
  { icon: '🐙', label: 'GitHub', value: 'github.com/yourusername', href: 'https://github.com/yourusername' },
  { icon: '💼', label: 'LinkedIn', value: 'linkedin.com/in/yourusername', href: 'https://linkedin.com/in/yourusername' },
];

/* ───────────────────────────────────────────
   TERMINAL FILE SYSTEM
   ─────────────────────────────────────────── */

const FILE_SYSTEM = {
  '/': {
    'about.txt': `Name: ${PROFILE.name}\nRole: ${PROFILE.title}\n\n${PROFILE.tagline}\n\nSkills: ${SKILLS.join(', ')}`,
    'projects.txt': PROJECTS.map(
      (p, i) => `${i + 1}. ${p.title}\n   ${p.description}\n   Tech: ${p.tech.join(', ')}`
    ).join('\n\n'),
    'contact.txt': CONTACT_LINKS.map((c) => `${c.label}: ${c.value}`).join('\n'),
    'resume.txt':
      'Education: B.S. Computer Science\nExperience: 3+ years of full-stack development\n\nDownload my full resume at: https://example.com/resume.pdf',
  },
};

/* ───────────────────────────────────────────
   TERMINAL COMMANDS
   ─────────────────────────────────────────── */

const buildCommands = (addOutput, setHistory) => ({
  help: () =>
    'Available commands:\n  help      — Show this message\n  ls        — List files\n  cat <f>   — Read a file\n  about     — About me\n  projects  — View projects\n  contact   — Contact info\n  clear     — Clear terminal\n  whoami    — Who am I?\n  date      — Current date',
  ls: () => Object.keys(FILE_SYSTEM['/']).join('  '),
  cat: (args) => {
    const file = args[0];
    if (!file) return 'Usage: cat <filename>';
    if (FILE_SYSTEM['/'][file]) return FILE_SYSTEM['/'][file];
    return `cat: ${file}: No such file or directory`;
  },
  about: () => FILE_SYSTEM['/']['about.txt'],
  projects: () => FILE_SYSTEM['/']['projects.txt'],
  contact: () => FILE_SYSTEM['/']['contact.txt'],
  clear: () => {
    setHistory([]);
    return null;
  },
  whoami: () => `${PROFILE.name} — ${PROFILE.title}`,
  date: () => new Date().toString(),
});

/* ───────────────────────────────────────────
   UTILITY HOOKS
   ─────────────────────────────────────────── */

function useClock() {
  const [time, setTime] = useState(new Date());
  useEffect(() => {
    const id = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return time;
}

/* ───────────────────────────────────────────
   WINDOW COMPONENT
   ─────────────────────────────────────────── */

function Window({ id, title, icon, children, isActive, onFocus, onClose, onMaximize, isMaximized, isMobile }) {
  const [pos, setPos] = useState({ x: 80 + id * 30, y: 60 + id * 30 });
  const [size, setSize] = useState({ w: 620, h: 440 });
  const dragRef = useRef(null);
  const resizeRef = useRef(null);

  /* Drag */
  const onDragStart = useCallback(
    (e) => {
      if (isMobile || isMaximized) return;
      e.preventDefault();
      onFocus();
      const startX = e.clientX - pos.x;
      const startY = e.clientY - pos.y;
      const onMove = (ev) => setPos({ x: ev.clientX - startX, y: Math.max(32, ev.clientY - startY) });
      const onUp = () => {
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
      };
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
    },
    [isMobile, isMaximized, onFocus, pos]
  );

  /* Resize */
  const onResizeStart = useCallback(
    (e) => {
      if (isMobile || isMaximized) return;
      e.preventDefault();
      e.stopPropagation();
      onFocus();
      const startX = e.clientX;
      const startY = e.clientY;
      const startW = size.w;
      const startH = size.h;
      const onMove = (ev) =>
        setSize({
          w: Math.max(360, startW + (ev.clientX - startX)),
          h: Math.max(260, startH + (ev.clientY - startY)),
        });
      const onUp = () => {
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
      };
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
    },
    [isMobile, isMaximized, onFocus, size]
  );

  const style =
    isMobile || isMaximized
      ? { top: 32, left: 0, width: '100%', height: 'calc(100vh - 32px - 80px)', borderRadius: 0 }
      : { top: pos.y, left: pos.x, width: size.w, height: size.h };

  return (
    <div
      className={`window ${isActive ? 'window--active' : ''}`}
      style={{ ...style, zIndex: isActive ? 100 : 10 }}
      onMouseDown={onFocus}
    >
      {/* Title bar */}
      <div className="window__titlebar" onMouseDown={onDragStart} ref={dragRef}>
        <div className="window__traffic">
          <button className="window__btn window__btn--close" onClick={onClose} title="Close" />
          <button className="window__btn window__btn--min" title="Minimize" />
          <button className="window__btn window__btn--max" onClick={onMaximize} title="Maximize" />
        </div>
        <span className="window__title">
          {icon} {title}
        </span>
        <div className="window__traffic-spacer" />
      </div>

      {/* Body */}
      <div className="window__body">{children}</div>

      {/* Resize handle */}
      {!isMobile && !isMaximized && (
        <div className="window__resize" onMouseDown={onResizeStart} ref={resizeRef} />
      )}
    </div>
  );
}

/* ───────────────────────────────────────────
   APP CONTENT PANELS
   ─────────────────────────────────────────── */

function AboutPanel() {
  return (
    <div className="panel panel--about">
      <h2>{PROFILE.name}</h2>
      <p className="panel__subtitle">{PROFILE.title}</p>
      <p className="panel__tagline">{PROFILE.tagline}</p>

      <div className="info-grid">
        {ABOUT_CARDS.map((c) => (
          <div className="info-card" key={c.label}>
            <span className="info-card__label">{c.label}</span>
            <span className="info-card__value">{c.value}</span>
          </div>
        ))}
      </div>

      <h3>Skills</h3>
      <div className="skill-tags">
        {SKILLS.map((s) => (
          <span className="skill-tag" key={s}>{s}</span>
        ))}
      </div>
    </div>
  );
}

function ProjectsPanel() {
  return (
    <div className="panel panel--projects">
      <h2>Projects</h2>
      <div className="project-grid">
        {PROJECTS.map((p) => (
          <div className="project-card" key={p.title}>
            <h4>{p.title}</h4>
            <p>{p.description}</p>
            <div className="project-card__tags">
              {p.tech.map((t) => (
                <span key={t} className="project-card__tag">{t}</span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ContactPanel() {
  return (
    <div className="panel panel--contact">
      <h2>Get In Touch</h2>
      <p className="panel__subtitle">I'd love to hear from you!</p>
      <div className="contact-list">
        {CONTACT_LINKS.map((c) => (
          <a href={c.href} className="contact-item" key={c.label} target="_blank" rel="noopener noreferrer">
            <span className="contact-item__icon">{c.icon}</span>
            <div>
              <span className="contact-item__label">{c.label}</span>
              <span className="contact-item__value">{c.value}</span>
            </div>
          </a>
        ))}
      </div>
    </div>
  );
}

function TerminalPanel() {
  const [history, setHistory] = useState([
    { type: 'output', text: `Welcome to Portfolio OS Terminal\nType "help" for available commands.\n` },
  ]);
  const [input, setInput] = useState('');
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [history]);

  const addOutput = (text) => setHistory((h) => [...h, { type: 'output', text }]);

  const commands = buildCommands(addOutput, setHistory);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!input.trim()) return;
    const trimmed = input.trim();
    const parts = trimmed.split(/\s+/);
    const cmd = parts[0].toLowerCase();
    const args = parts.slice(1);

    setHistory((h) => [...h, { type: 'input', text: trimmed }]);

    if (commands[cmd]) {
      const result = commands[cmd](args);
      if (result !== null && result !== undefined) addOutput(result);
    } else {
      addOutput(`command not found: ${cmd}. Type "help" for available commands.`);
    }
    setInput('');
  };

  return (
    <div className="terminal" ref={scrollRef}>
      <div className="terminal__history">
        {history.map((entry, i) => (
          <div key={i} className={`terminal__line terminal__line--${entry.type}`}>
            {entry.type === 'input' && <span className="terminal__prompt">❯ </span>}
            <pre>{entry.text}</pre>
          </div>
        ))}
      </div>
      <form className="terminal__input-row" onSubmit={handleSubmit}>
        <span className="terminal__prompt">❯ </span>
        <input
          className="terminal__input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          autoFocus
          spellCheck={false}
          placeholder="Type a command…"
        />
      </form>
    </div>
  );
}

/* ───────────────────────────────────────────
   APP DEFINITIONS
   ─────────────────────────────────────────── */

const APPS = [
  { name: 'About', icon: '👤', Component: AboutPanel },
  { name: 'Projects', icon: '💼', Component: ProjectsPanel },
  { name: 'Terminal', icon: '⌘', Component: TerminalPanel },
  { name: 'Contact', icon: '📧', Component: ContactPanel },
];

/* ───────────────────────────────────────────
   MAIN APP
   ─────────────────────────────────────────── */

export default function App() {
  const clock = useClock();
  const [openWindows, setOpenWindows] = useState([]); // array of { id, appIndex }
  const [activeId, setActiveId] = useState(null);
  const [maximized, setMaximized] = useState({});
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);
  const nextId = useRef(1);

  useEffect(() => {
    const handler = () => setIsMobile(window.innerWidth < 768);
    window.addEventListener('resize', handler);
    return () => window.removeEventListener('resize', handler);
  }, []);

  const openApp = (appIndex) => {
    const id = nextId.current++;
    setOpenWindows((w) => [...w, { id, appIndex }]);
    setActiveId(id);
  };

  const closeWindow = (id) => {
    setOpenWindows((w) => w.filter((win) => win.id !== id));
    setMaximized((m) => {
      const copy = { ...m };
      delete copy[id];
      return copy;
    });
    setActiveId((prev) => {
      if (prev !== id) return prev;
      const remaining = openWindows.filter((w) => w.id !== id);
      return remaining.length ? remaining[remaining.length - 1].id : null;
    });
  };

  const toggleMaximize = (id) => setMaximized((m) => ({ ...m, [id]: !m[id] }));

  const formattedTime = clock.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  const formattedDate = clock.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });

  return (
    <div className="os">
      {/* ─── Menu Bar ─── */}
      <header className="menubar">
        <div className="menubar__left">
          <span className="menubar__logo">&#63743;</span>
          <span className="menubar__item">{PROFILE.name}</span>
        </div>
        <div className="menubar__right">
          <span className="menubar__item">{formattedDate}</span>
          <span className="menubar__item menubar__time">{formattedTime}</span>
        </div>
      </header>

      {/* ─── Desktop ─── */}
      <main className="desktop">
        {openWindows.length === 0 && (
          <div className="desktop__welcome">
            <h1>{PROFILE.name}</h1>
            <p>{PROFILE.tagline}</p>
            <p className="desktop__hint">Click an icon in the dock below to get started.</p>
          </div>
        )}

        {openWindows.map((win) => {
          const app = APPS[win.appIndex];
          return (
            <Window
              key={win.id}
              id={win.id}
              title={app.name}
              icon={app.icon}
              isActive={activeId === win.id}
              onFocus={() => setActiveId(win.id)}
              onClose={() => closeWindow(win.id)}
              onMaximize={() => toggleMaximize(win.id)}
              isMaximized={!!maximized[win.id]}
              isMobile={isMobile}
            >
              <app.Component />
            </Window>
          );
        })}
      </main>

      {/* ─── Dock ─── */}
      <nav className="dock">
        <div className="dock__bar">
          {APPS.map((app, i) => (
            <button key={app.name} className="dock__item" onClick={() => openApp(i)} title={app.name}>
              <span className="dock__icon">{app.icon}</span>
              <span className="dock__label">{app.name}</span>
            </button>
          ))}
        </div>
      </nav>
    </div>
  );
}
