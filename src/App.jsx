import React, { useState, useRef, useEffect, useCallback } from 'react';
import './App.css';

/* ============================================================
   1.  EDIT YOUR CONTENT HERE
   ============================================================ */

const PROFILE = {
  name:   'Jose Garcia',
  handle: 'jgarc521',
  role:   'Software Engineer',
};

const ABOUT_CARDS = [
  { k: 'Location', v: 'California, USA' },
  { k: 'Focus',    v: 'Full-stack web' },
  { k: 'Status',   v: 'Open to work' },
  { k: 'Coffee',   v: 'Oat flat white' },
];

const ABOUT_TEXT = [
  `Hi — I'm ${PROFILE.name}, a ${PROFILE.role.toLowerCase()} who likes turning messy problems into tidy software.`,
  `Most of my time goes into shipping reliable backends and front-ends that don't make people think too hard. This whole site is one of those experiments: a tiny desktop you can poke at.`,
];

const SKILLS = ['JavaScript', 'TypeScript', 'React', 'Node.js', 'Python', 'PostgreSQL', 'Git', 'AWS', 'Docker', 'REST APIs'];

const PROJECTS = [
  { name: 'Project One', year: '2026', desc: 'A short, honest sentence about what it does and why it mattered.',
    links: [{ label: 'Live', href: 'https://example.com' }, { label: 'GitHub', href: 'https://github.com/jgarc521' }] },
  { name: 'Project Two', year: '2025', desc: 'Another one. Lead with the impact or the interesting technical bit.',
    links: [{ label: 'GitHub', href: 'https://github.com/jgarc521' }] },
  { name: 'This Portfolio', year: '2026', desc: 'A desktop-style portfolio built in React. The thing you are using right now.',
    links: [{ label: 'GitHub', href: 'https://github.com/jgarc521' }] },
];

const CONTACT_LINKS = [
  { what: 'email',    where: 'you@example.com',          href: 'mailto:you@example.com',           icon: 'mail' },
  { what: 'github',   where: 'github.com/jgarc521',        href: 'https://github.com/jgarc521',      icon: 'github' },
  { what: 'linkedin', where: 'linkedin.com/in/jgarc521',   href: 'https://linkedin.com/in/jgarc521', icon: 'linkedin' },
];

/* DESKTOP SHORTCUTS — documents only. Put the files in /public.
   web: false  → opens in the in-OS viewer (good for your own PDFs)
   web: true   → opens in a new browser tab (good for external links) */
const PUBLIC = process.env.PUBLIC_URL || '';
const RESUME_URL = `${PUBLIC}/resume.pdf`;

const DOCUMENTS = [
  { name: 'Résumé',       ext: 'PDF', icon: 'file',  href: RESUME_URL,                   web: false },
  { name: 'Transcript',   ext: 'PDF', icon: 'file',  href: `${PUBLIC}/transcript.pdf`,   web: false },
  { name: 'Cover Letter', ext: 'PDF', icon: 'file',  href: `${PUBLIC}/cover-letter.pdf`, web: false },
  { name: 'Certs',        ext: 'WEB', icon: 'badge', href: 'https://example.com/certs',  web: true  },
];

const FILE_SYSTEM = {
  'about.txt':   ABOUT_TEXT.join('\n\n'),
  'skills.txt':  SKILLS.join(', '),
  'contact.txt': CONTACT_LINKS.map(c => `${c.what}: ${c.where}`).join('\n'),
  'resume.pdf':  `[binary] open it with the Résumé icon on the desktop, or type "resume".`,
};

const SWATCHES = [
  { name: 'Steel',   a: '#4682b4', s: '#6ba3d6' },
  { name: 'Emerald', a: '#3fa17a', s: '#5fc79b' },
  { name: 'Amber',   a: '#cf9239', s: '#e6b260' },
  { name: 'Violet',  a: '#8a7bd8', s: '#a99cf0' },
  { name: 'Rose',    a: '#d4688a', s: '#ee8aaa' },
  { name: 'Cyan',    a: '#3fa6bd', s: '#5fc7dd' },
];

/* ============================================================
   2.  ICONS
   ============================================================ */
const Icon = ({ name }) => {
  const p = { fill: 'none', stroke: 'currentColor', strokeWidth: 1.7, strokeLinecap: 'round', strokeLinejoin: 'round' };
  const paths = {
    user:     <><circle cx="12" cy="8" r="3.4" {...p}/><path d="M5 20c0-3.6 3.1-5.6 7-5.6s7 2 7 5.6" {...p}/></>,
    folder:   <><path d="M3 7.5A1.5 1.5 0 0 1 4.5 6H9l2 2h8.5A1.5 1.5 0 0 1 21 9.5V18a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 18V7.5Z" {...p}/></>,
    file:     <><path d="M6 3h8l4 4v14H6z" {...p}/><path d="M14 3v4h4" {...p}/><path d="M9 13h6M9 16h6" {...p}/></>,
    spark:    <><path d="M12 3v18M3 12h18M6 6l12 12M18 6 6 18" {...p}/></>,
    mail:     <><rect x="3" y="5" width="18" height="14" rx="2" {...p}/><path d="m4 7 8 6 8-6" {...p}/></>,
    terminal: <><rect x="3" y="4" width="18" height="16" rx="2" {...p}/><path d="m7 9 3 3-3 3M13 15h4" {...p}/></>,
    game:     <><rect x="2.5" y="7.5" width="19" height="10" rx="4.5" {...p}/><path d="M7 11.5v3M5.5 13h3M15.5 12h.01M18 13.5h.01M16.5 15h.01" {...p}/></>,
    badge:    <><circle cx="12" cy="9" r="5" {...p}/><path d="M8.5 13 7 21l5-2.5L17 21l-1.5-8" {...p}/></>,
    github:   <><path d="M9 19c-4 1.3-4-2-6-2.5m12 4.5v-3.2c0-.9.1-1.3-.4-1.8 2.4-.3 4.9-1.2 4.9-5.3a4 4 0 0 0-1.1-2.8 3.7 3.7 0 0 0-.1-2.8s-.9-.3-3 1.1a10 10 0 0 0-5.4 0C7.8 3.7 6.9 4 6.9 4a3.7 3.7 0 0 0-.1 2.8A4 4 0 0 0 5.7 9.6c0 4 2.5 5 4.9 5.3-.4.4-.4.9-.4 1.7V21" {...p}/></>,
    linkedin: <><rect x="3" y="3" width="18" height="18" rx="2" {...p}/><path d="M8 10v6M8 7v.01M12 16v-3.5a1.5 1.5 0 0 1 3 0V16M12 13v3" {...p}/></>,
  };
  return <svg viewBox="0 0 24 24" aria-hidden="true">{paths[name] || paths.file}</svg>;
};

/* ============================================================
   3.  APP PANELS
   ============================================================ */
function AboutPanel() {
  return (
    <div>
      <p className="kicker">whoami</p>
      <h2 className="h1">{PROFILE.name}</h2>
      <p className="lead">{PROFILE.role}</p>
      {ABOUT_TEXT.map((t, i) => <p className="p" key={i}>{t}</p>)}
      <div className="cards">
        {ABOUT_CARDS.map(c => (<div className="card" key={c.k}><div className="card-k">{c.k}</div><div className="card-v">{c.v}</div></div>))}
      </div>
    </div>
  );
}
function SkillsPanel() {
  return (
    <div>
      <p className="kicker">stack</p>
      <h2 className="h1">Tools I reach for</h2>
      <p className="lead">The things I'm comfortable building production work with.</p>
      <div className="tag-row">{SKILLS.map(s => <span className="tag" key={s}>{s}</span>)}</div>
    </div>
  );
}
function ProjectsPanel() {
  return (
    <div>
      <p className="kicker">selected work</p>
      <h2 className="h1">Projects</h2>
      <p className="lead">A few things worth showing.</p>
      {PROJECTS.map(pr => (
        <div className="proj" key={pr.name}>
          <div className="proj-top"><span className="proj-name">{pr.name}</span><span className="proj-year">{pr.year}</span></div>
          <p className="proj-desc">{pr.desc}</p>
          <div className="proj-links">{pr.links.map(l => <a className="link" key={l.label} href={l.href} target="_blank" rel="noreferrer">{l.label} ↗</a>)}</div>
        </div>
      ))}
    </div>
  );
}
function ContactPanel() {
  return (
    <div>
      <p className="kicker">say hi</p>
      <h2 className="h1">Get in touch</h2>
      <p className="lead">The fastest ways to reach me.</p>
      {CONTACT_LINKS.map(c => (
        <a className="contact-row" key={c.what} href={c.href} target="_blank" rel="noreferrer">
          <span className="contact-ic"><Icon name={c.icon} /></span>
          <span><div className="what">{c.what}</div><div className="where">{c.where}</div></span>
        </a>
      ))}
    </div>
  );
}

function ViewerPanel({ data }) {
  const doc = (data && data.doc) || {};
  return (
    <div className="viewer">
      <div className="viewer-bar">
        <span className="viewer-name">{doc.name}</span>
        <span className="viewer-actions">
          <a className="btn ghost" href={doc.href} target="_blank" rel="noreferrer">Open in new tab ↗</a>
          <a className="btn" href={doc.href} download>Download</a>
        </span>
      </div>
      <iframe className="viewer-frame" src={doc.href} title={doc.name} />
      <span className="viewer-hint">If the file looks blank, it isn't in /public yet — add {doc.href}</span>
    </div>
  );
}

function TerminalPanel() {
  const [history, setHistory] = useState([{ type: 'sys', text: `Welcome. Type "help" to begin.` }]);
  const [value, setValue] = useState('');
  const endRef = useRef(null);
  const inputRef = useRef(null);
  useEffect(() => { endRef.current?.scrollIntoView(); }, [history]);
  const run = (raw) => {
    const cmd = raw.trim();
    const [name, ...args] = cmd.split(/\s+/);
    const out = [];
    const say = (text, cls = 'out') => out.push({ type: cls, text });
    switch (name.toLowerCase()) {
      case '': break;
      case 'help':     say('available: about · skills · projects · resume · contact · ls · cat <file> · neofetch · date · clear'); break;
      case 'about':    say(ABOUT_TEXT.join('\n\n')); break;
      case 'skills':   say(SKILLS.join(', ')); break;
      case 'projects': PROJECTS.forEach(pr => say(`${pr.name} (${pr.year}) — ${pr.desc}`)); break;
      case 'contact':  CONTACT_LINKS.forEach(c => say(`${c.what}: ${c.where}`)); break;
      case 'resume':   window.open(RESUME_URL, '_blank'); say('opening resume.pdf...', 'ok'); break;
      case 'ls':       say(Object.keys(FILE_SYSTEM).join('   ')); break;
      case 'cat': { const f = args[0]; say(FILE_SYSTEM[f] != null ? FILE_SYSTEM[f] : `cat: ${f || ''}: no such file`); break; }
      case 'whoami':   say(PROFILE.handle); break;
      case 'date':     say(new Date().toString()); break;
      case 'neofetch': say(`${PROFILE.handle}@web`, 'ok'); say(`host     ${PROFILE.role}`); say(`shell    portfolio-sh`); say(`uptime   online`); break;
      case 'clear':    setHistory([]); setValue(''); return;
      default:         say(`command not found: ${name}. try "help".`);
    }
    setHistory(h => [...h, { type: 'cmd', text: cmd }, ...out]);
    setValue('');
  };
  return (
    <div className="term" onClick={() => inputRef.current?.focus()}>
      <div className="term-out">
        {history.map((h, i) => (
          <div key={i} className={h.type === 'ok' ? 'ok' : undefined}>
            {h.type === 'cmd' ? <span className="cmd">{PROFILE.handle}$ {h.text}</span> : h.text}
          </div>
        ))}
        <div ref={endRef} />
      </div>
      <div className="term-prompt">
        <span className="ps1">{PROFILE.handle}$</span>
        <input ref={inputRef} className="term-input" value={value} autoFocus aria-label="terminal input"
          onChange={e => setValue(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') run(value); }} />
      </div>
    </div>
  );
}

/* ============================================================
   4.  GAMES
   ============================================================ */
const TTT_LINES = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];
function TicTacToe() {
  const [b, setB] = useState(Array(9).fill(null));
  const [result, setResult] = useState(null);      // {who} | {draw} | null
  const [winLine, setWinLine] = useState([]);
  const [tally, setTally] = useState({ X: 0, O: 0 });
  const check = (bd) => {
    for (const ln of TTT_LINES) { const [a, c, d] = ln; if (bd[a] && bd[a] === bd[c] && bd[a] === bd[d]) return { who: bd[a], line: ln }; }
    return bd.every(Boolean) ? { draw: true } : null;
  };
  const play = (i) => {
    if (b[i] || result) return;
    const nb = b.slice(); nb[i] = 'X';
    let r = check(nb);
    if (!r) {
      const empties = nb.map((v, idx) => v ? null : idx).filter(v => v != null);
      if (empties.length) { nb[empties[Math.floor(Math.random() * empties.length)]] = 'O'; r = check(nb); }
    }
    setB(nb);
    if (r) { setResult(r); setWinLine(r.line || []); if (r.who) setTally(t => ({ ...t, [r.who]: t[r.who] + 1 })); }
  };
  const reset = () => { setB(Array(9).fill(null)); setResult(null); setWinLine([]); };
  const status = result ? (result.draw ? 'Draw' : (result.who === 'X' ? 'You win' : 'CPU wins')) : 'Your move (X)';
  return (
    <div className="game">
      <div className="score"><span>You <b>{tally.X}</b></span><span>CPU <b>{tally.O}</b></span></div>
      <div className="ttt">
        {b.map((v, i) => (
          <button key={i} className={`ttt-cell ${v === 'X' ? 'x' : v === 'O' ? 'o' : ''} ${winLine.includes(i) ? 'win' : ''}`} onClick={() => play(i)}>{v}</button>
        ))}
      </div>
      <div className="game-bar"><span className="game-status">{status}</span><button className="btn ghost" onClick={reset}>Reset board</button></div>
    </div>
  );
}

const MEM_EMOJI = ['🌵','🚀','🎲','🐙','🍣','⚡','🎧','🛰️'];
function Memory() {
  const make = () => {
    const cards = [...MEM_EMOJI, ...MEM_EMOJI];
    for (let j = cards.length - 1; j > 0; j--) { const k = Math.floor(Math.random() * (j + 1)); [cards[j], cards[k]] = [cards[k], cards[j]]; }
    return cards.map((e, id) => ({ id, e, done: false }));
  };
  const [cards, setCards] = useState(make);
  const [open, setOpen] = useState([]);
  const [moves, setMoves] = useState(0);
  const [lock, setLock] = useState(false);
  const flip = (idx) => {
    if (lock || cards[idx].done || open.includes(idx)) return;
    const no = [...open, idx];
    setOpen(no);
    if (no.length === 2) {
      setMoves(m => m + 1); setLock(true);
      const [a, c] = no;
      if (cards[a].e === cards[c].e) setTimeout(() => { setCards(cs => cs.map((card, i) => (i === a || i === c) ? { ...card, done: true } : card)); setOpen([]); setLock(false); }, 420);
      else setTimeout(() => { setOpen([]); setLock(false); }, 720);
    }
  };
  const reset = () => { setCards(make()); setOpen([]); setMoves(0); setLock(false); };
  const won = cards.every(c => c.done);
  return (
    <div className="game">
      <div className="mem">
        {cards.map((c, i) => {
          const show = c.done || open.includes(i);
          return <button key={c.id} className={`mem-card ${show ? 'show' : ''} ${c.done ? 'done' : ''}`} onClick={() => flip(i)}>{show ? c.e : ''}</button>;
        })}
      </div>
      <div className="game-bar"><span className="game-status">{won ? `Solved in ${moves} moves 🎉` : `Moves: ${moves}`}</span><button className="btn ghost" onClick={reset}>New game</button></div>
    </div>
  );
}

function Snake() {
  const SIZE = 15;
  const [snake, setSnake] = useState([{ x: 7, y: 7 }]);
  const [food, setFood] = useState({ x: 4, y: 4 });
  const [running, setRunning] = useState(false);
  const [over, setOver] = useState(false);
  const [score, setScore] = useState(0);
  const dirRef = useRef({ x: 1, y: 0 });
  const snakeRef = useRef(snake); snakeRef.current = snake;
  const foodRef = useRef(food); foodRef.current = food;
  const place = (sn) => { let f; do { f = { x: Math.floor(Math.random() * SIZE), y: Math.floor(Math.random() * SIZE) }; } while (sn.some(s => s.x === f.x && s.y === f.y)); return f; };
  const setD = (d) => { const c = dirRef.current; if (d.x === -c.x && d.y === -c.y) return; dirRef.current = d; };
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      const c = dirRef.current, sn = snakeRef.current;
      const head = { x: sn[0].x + c.x, y: sn[0].y + c.y };
      if (head.x < 0 || head.y < 0 || head.x >= SIZE || head.y >= SIZE || sn.some(s => s.x === head.x && s.y === head.y)) { setRunning(false); setOver(true); return; }
      let ns = [head, ...sn]; const fd = foodRef.current;
      if (head.x === fd.x && head.y === fd.y) { setScore(s => s + 1); setFood(place(ns)); } else ns = ns.slice(0, -1);
      setSnake(ns);
    }, 140);
    return () => clearInterval(id);
  }, [running]);
  useEffect(() => {
    const onKey = (e) => {
      if (/INPUT|TEXTAREA/.test(e.target.tagName)) return;
      const m = { ArrowUp:{x:0,y:-1}, ArrowDown:{x:0,y:1}, ArrowLeft:{x:-1,y:0}, ArrowRight:{x:1,y:0}, w:{x:0,y:-1}, s:{x:0,y:1}, a:{x:-1,y:0}, d:{x:1,y:0} };
      const nd = m[e.key]; if (nd) { e.preventDefault(); if (!running && !over) setRunning(true); setD(nd); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [running, over]);
  const start = () => { const s = [{ x: 7, y: 7 }]; setSnake(s); dirRef.current = { x: 1, y: 0 }; setFood(place(s)); setScore(0); setOver(false); setRunning(true); };
  const nudge = (d) => { if (!running && !over) setRunning(true); setD(d); };
  const cells = [];
  for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) {
    const isHead = snake[0].x === x && snake[0].y === y;
    const isBody = snake.some((s, i) => i > 0 && s.x === x && s.y === y);
    const isFood = food.x === x && food.y === y;
    cells.push(<div key={`${x}-${y}`} className={`sn-cell ${isHead ? 'head' : ''} ${isBody ? 'body' : ''} ${isFood ? 'food' : ''}`} />);
  }
  return (
    <div className="game">
      <div className="sn-wrap">
        <div className="sn-board" style={{ gridTemplateColumns: `repeat(${SIZE},1fr)`, gridTemplateRows: `repeat(${SIZE},1fr)` }}>{cells}</div>
        {!running && <button className="sn-overlay" onClick={start}>{over ? `Game over · score ${score} · play again` : 'Tap, or press an arrow / WASD to start'}</button>}
      </div>
      <div className="game-bar">
        <span className="game-status">Score: {score}</span>
        <div className="dpad">
          <button className="up" onClick={() => nudge({ x:0, y:-1 })} aria-label="up">↑</button>
          <button className="left" onClick={() => nudge({ x:-1, y:0 })} aria-label="left">←</button>
          <button className="down" onClick={() => nudge({ x:0, y:1 })} aria-label="down">↓</button>
          <button className="right" onClick={() => nudge({ x:1, y:0 })} aria-label="right">→</button>
        </div>
      </div>
    </div>
  );
}

const GAME_LIST = [
  { id: 'ttt',   name: 'Tic-Tac-Toe', C: TicTacToe },
  { id: 'mem',   name: 'Memory',      C: Memory },
  { id: 'snake', name: 'Snake',       C: Snake },
];
function GamesPanel() {
  const [g, setG] = useState('ttt');
  const Active = GAME_LIST.find(x => x.id === g).C;
  return (
    <div>
      <p className="kicker">arcade</p>
      <h2 className="h1">Games</h2>
      <div className="gtabs">{GAME_LIST.map(x => <button key={x.id} className={`gtab ${g === x.id ? 'active' : ''}`} onClick={() => setG(x.id)}>{x.name}</button>)}</div>
      <div style={{ marginTop: 18 }}><Active /></div>
    </div>
  );
}

/* ============================================================
   5.  APP REGISTRY  (hidden:true = launchable but not in dock)
   ============================================================ */
const APPS = [
  { id: 'about',    name: 'About',    icon: 'user',     w: 480, h: 470, Component: AboutPanel },
  { id: 'projects', name: 'Projects', icon: 'folder',   w: 540, h: 520, Component: ProjectsPanel },
  { id: 'skills',   name: 'Skills',   icon: 'spark',    w: 460, h: 340, Component: SkillsPanel },
  { id: 'contact',  name: 'Contact',  icon: 'mail',     w: 440, h: 360, Component: ContactPanel },
  { id: 'games',    name: 'Games',    icon: 'game',     w: 520, h: 600, minH: 540, Component: GamesPanel },
  { id: 'terminal', name: 'Terminal', icon: 'terminal', w: 600, h: 440, minW: 380, minH: 300, Component: TerminalPanel },
  { id: 'viewer',   name: 'Viewer',   icon: 'file',     w: 760, h: 600, minW: 360, minH: 300, Component: ViewerPanel, hidden: true, multi: true },
];

/* ============================================================
   6.  BOOT SCREEN
   ============================================================ */
const BOOT_LINES = [
  { t: 'booting…', head: true },
  { t: 'POST .................................', ok: true },
  { t: 'Loading kernel modules .............', ok: true },
  { t: `Mounting /home/${PROFILE.handle} ........`, ok: true },
  { t: 'Starting window manager ............', ok: true },
  { t: 'Launching services .................', ok: true },
];
function BootScreen({ onDone }) {
  const [shown, setShown] = useState(0);
  const [progress, setProgress] = useState(0);
  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) { onDone(); return; }
    const timers = [];
    BOOT_LINES.forEach((_, i) => timers.push(setTimeout(() => setShown(i + 1), 320 + i * 360)));
    const total = 320 + BOOT_LINES.length * 360 + 500;
    const start = Date.now();
    const tick = setInterval(() => { const pct = Math.min(100, ((Date.now() - start) / total) * 100); setProgress(pct); if (pct >= 100) clearInterval(tick); }, 60);
    timers.push(setTimeout(onDone, total + 450));
    return () => { timers.forEach(clearTimeout); clearInterval(tick); };
  }, [onDone]);
  return (
    <div className="boot" onClick={onDone} role="status" aria-label="System booting">
      {BOOT_LINES.slice(0, shown).map((l, i) => (
        <div className={`boot-line ${l.head ? 'boot-head' : ''}`} key={i}>{l.ok ? <>{l.t} <span className="boot-ok">[ OK ]</span></> : l.t}</div>
      ))}
      {shown < BOOT_LINES.length && <span className="boot-cursor" />}
      <div className="boot-bar-wrap">
        <div className="boot-bar-label"><span>starting desktop…</span><span>{Math.round(progress)}%</span></div>
        <div className="boot-bar"><div className="boot-bar-fill" style={{ width: `${progress}%` }} /></div>
        <button className="boot-skip" onClick={onDone}>skip ⏎</button>
      </div>
    </div>
  );
}

/* ============================================================
   7.  WINDOW  (draggable + resizable + maximize)
   Gesture handlers are created ONCE and kept stable so a focus
   re-render can never tear them off mid-drag.
   ============================================================ */
function Win({ data, onFocus, onClose, onMinimize, onMaximize, onCommit }) {
  const app = APPS.find(a => a.id === data.app);
  const ref = useRef(null);
  const gesture = useRef(null);
  const commitRef = useRef(onCommit); commitRef.current = onCommit;
  const idRef = useRef(data.id);
  const h = useRef(null);
  if (!h.current) {
    const move = (e) => {
      const g = gesture.current; if (!g) return;
      const el = ref.current; if (!el) return;
      if (g.type === 'move') {
        const x = Math.max(2, Math.min(window.innerWidth - 120, e.clientX - g.dx));
        const y = Math.max(2, Math.min(window.innerHeight - 80, e.clientY - g.dy));
        el.style.left = x + 'px'; el.style.top = y + 'px';
      } else {
        let w = Math.max(g.minW, g.startW + (e.clientX - g.sx));
        let ht = Math.max(g.minH, g.startH + (e.clientY - g.sy));
        w = Math.min(w, window.innerWidth - el.offsetLeft - 6);
        ht = Math.min(ht, window.innerHeight - el.offsetTop - 6);
        el.style.width = w + 'px'; el.style.height = ht + 'px';
      }
    };
    const up = () => {
      if (!gesture.current) return;
      gesture.current = null;
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      const r = ref.current.getBoundingClientRect();
      commitRef.current(idRef.current, { x: Math.round(r.left), y: Math.round(r.top), w: Math.round(r.width), h: Math.round(r.height) });
    };
    h.current = { move, up };
  }
  const { move, up } = h.current;
  useEffect(() => () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); }, [move, up]);

  const startMove = (e) => {
    onFocus(data.id);
    if (e.target.closest('.win-ctrl') || e.target.closest('.win-resize')) return;
    const r = ref.current.getBoundingClientRect();
    gesture.current = { type: 'move', dx: e.clientX - r.left, dy: e.clientY - r.top };
    window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
  };
  const startResize = (e) => {
    e.stopPropagation(); onFocus(data.id);
    const r = ref.current.getBoundingClientRect();
    gesture.current = { type: 'resize', sx: e.clientX, sy: e.clientY, startW: r.width, startH: r.height, minW: app.minW || 300, minH: app.minH || 220 };
    window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
  };

  if (data.minimized) return null;
  const Body = app.Component;
  return (
    <div ref={ref} className="win" style={{ left: data.x, top: data.y, width: data.w, height: data.h, zIndex: data.z }} onMouseDown={() => onFocus(data.id)}>
      <div className="win-bar" onPointerDown={startMove} onDoubleClick={() => onMaximize(data.id)}>
        <span className="win-dot" />
        <span className="win-title">{data.title || app.name}</span>
        <div className="win-ctrls">
          <button className="win-ctrl" aria-label="Minimize" onClick={() => onMinimize(data.id)}>—</button>
          <button className="win-ctrl" aria-label="Maximize" onClick={() => onMaximize(data.id)}>▢</button>
          <button className="win-ctrl close" aria-label="Close" onClick={() => onClose(data.id)}>✕</button>
        </div>
      </div>
      <div className="win-body"><Body data={data} /></div>
      <span className="win-resize" onPointerDown={startResize} />
    </div>
  );
}

/* ============================================================
   8.  CLOCK
   ============================================================ */
function Clock() {
  const [now, setNow] = useState(new Date());
  useEffect(() => { const t = setInterval(() => setNow(new Date()), 30000); return () => clearInterval(t); }, []);
  return (
    <div className="clock-float">
      <div>{now.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}</div>
      <div className="date">{now.toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric' })}</div>
    </div>
  );
}

/* ============================================================
   9.  APP ROOT
   ============================================================ */
export default function App() {
  const [phase, setPhase] = useState('boot');
  const [wins, setWins] = useState([]);
  const [marquee, setMarquee] = useState(null);
  const [selDocs, setSelDocs] = useState(() => new Set());
  const [ctx, setCtx] = useState(null);
  const z = useRef(10);
  const opened = useRef(0);
  const osRef = useRef(null);
  const iconRefs = useRef([]);
  const mq = useRef(null);

  const place = (app) => {
    const n = opened.current++; const cascade = (n % 5) * 32;
    const x = Math.max(120, Math.round(window.innerWidth / 2 - app.w / 2) - 30 + cascade);
    const y = 58 + cascade;
    return { x, y };
  };
  const openApp = (id, payload) => {
    setCtx(null);
    const app = APPS.find(a => a.id === id);
    if (payload || app.multi) {
      const { x, y } = place(app);
      setWins(prev => [...prev, { id: `${id}-${Date.now()}`, app: id, x, y, w: app.w, h: app.h, z: ++z.current, ...payload }]);
      return;
    }
    setWins(prev => {
      const existing = prev.find(w => w.app === id);
      if (existing) return prev.map(w => w.app === id ? { ...w, minimized: false, z: ++z.current } : w);
      const { x, y } = place(app);
      return [...prev, { id: `${id}-${Date.now()}`, app: id, x, y, w: app.w, h: app.h, z: ++z.current }];
    });
  };
  const openViewer = (doc) => openApp('viewer', { doc, title: doc.name });
  const openDocItem = (d) => d.web ? window.open(d.href, '_blank', 'noopener') : openViewer(d);
  const focusWin = (wid) => setWins(p => p.map(w => w.id === wid ? { ...w, z: ++z.current } : w));
  const closeWin = (wid) => setWins(p => p.filter(w => w.id !== wid));
  const minWin   = (wid) => setWins(p => p.map(w => w.id === wid ? { ...w, minimized: true } : w));
  const commit   = (wid, box) => setWins(p => p.map(w => w.id === wid ? { ...w, ...box, restore: null } : w));
  const maximize = (wid) => setWins(p => p.map(w => {
    if (w.id !== wid) return w;
    if (w.restore) return { ...w, ...w.restore, restore: null, z: ++z.current };
    return { ...w, restore: { x: w.x, y: w.y, w: w.w, h: w.h }, x: 8, y: 8, w: window.innerWidth - 16, h: window.innerHeight - 16 - 88, z: ++z.current };
  }));

  useEffect(() => { if (phase === 'desktop') setTimeout(() => openApp('about'), 250); /* eslint-disable-next-line */ }, [phase]);

  useEffect(() => {
    if (!ctx) return;
    const handler = (e) => { if (!e.target.closest('.ctx')) setCtx(null); };
    document.addEventListener('pointerdown', handler);
    return () => document.removeEventListener('pointerdown', handler);
  }, [ctx]);

  const mqMove = useCallback((e) => {
    const s = mq.current; if (!s) return;
    const x = Math.min(s.x, e.clientX), y = Math.min(s.y, e.clientY);
    const w = Math.abs(e.clientX - s.x), ht = Math.abs(e.clientY - s.y);
    setMarquee({ x, y, w, h: ht });
    const sel = new Set();
    iconRefs.current.forEach((el, i) => { if (!el) return; const r = el.getBoundingClientRect(); if (r.left < x + w && r.right > x && r.top < y + ht && r.bottom > y) sel.add(i); });
    setSelDocs(sel);
  }, []);
  const mqUp = useCallback(() => {
    mq.current = null; setMarquee(null); setSelDocs(new Set());
    window.removeEventListener('pointermove', mqMove); window.removeEventListener('pointerup', mqUp);
  }, [mqMove]);
  const onDesktopDown = (e) => {
    if (e.target !== osRef.current || e.button !== 0) return;
    setCtx(null);
    mq.current = { x: e.clientX, y: e.clientY };
    window.addEventListener('pointermove', mqMove); window.addEventListener('pointerup', mqUp);
  };

  if (phase === 'boot') return <BootScreen onDone={() => setPhase('desktop')} />;
  const dockApps = APPS.filter(a => !a.hidden);

  return (
    <div className="os os-enter" ref={osRef} onPointerDown={onDesktopDown}
      onContextMenu={(e) => { e.preventDefault(); setCtx({ x: Math.min(e.clientX, window.innerWidth - 220), y: Math.min(e.clientY, window.innerHeight - 260) }); }}>
      <Clock />

      <div className="icons">
        {DOCUMENTS.map((d, i) => (
          <button className={`icon ${selDocs.has(i) ? 'selected' : ''}`} key={d.name} ref={el => (iconRefs.current[i] = el)} onClick={() => openDocItem(d)}>
            <span className="icon-glyph"><Icon name={d.icon} /><span className="icon-ext">{d.ext}</span></span>
            <span className="icon-label">{d.name}</span>
          </button>
        ))}
      </div>

      {wins.map(w => (
        <Win key={w.id} data={w} onFocus={focusWin} onClose={closeWin} onMinimize={minWin} onMaximize={maximize} onCommit={commit} />
      ))}

      {marquee && <div className="marquee" style={{ left: marquee.x, top: marquee.y, width: marquee.w, height: marquee.h }} />}

      {ctx && (
        <div className="ctx" style={{ left: ctx.x, top: ctx.y }}>
          <button className="ctx-item" onClick={() => openApp('terminal')}><Icon name="terminal" /> Open Terminal</button>
          <button className="ctx-item" onClick={() => openApp('games')}><Icon name="game" /> Open Games</button>
          <div className="ctx-sep" />
          <div className="ctx-label">Accent</div>
          <div className="ctx-swatches">
            {SWATCHES.map(sw => (
              <button key={sw.name} className="swatch" title={sw.name} style={{ background: sw.a }}
                onClick={() => { document.documentElement.style.setProperty('--accent', sw.a); document.documentElement.style.setProperty('--accent-soft', sw.s); setCtx(null); }} />
            ))}
          </div>
          <div className="ctx-sep" />
          <button className="ctx-item" onClick={() => { setWins([]); setCtx(null); }}><Icon name="spark" /> Close all windows</button>
        </div>
      )}

      <div className="dock">
        {dockApps.map(a => {
          const open = wins.some(w => w.app === a.id);
          return (
            <button className="dock-item" key={a.id} onClick={() => openApp(a.id)} aria-label={a.name}>
              <span className="tip">{a.name}</span><Icon name={a.icon} />{open && <span className="dock-dot" />}
            </button>
          );
        })}
      </div>
    </div>
  );
}