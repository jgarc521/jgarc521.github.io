# Portfolio OS

A macOS-inspired interactive portfolio website built with React. Draggable windows, a working terminal, and a dock — all running in the browser.

![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)
![License](https://img.shields.io/badge/License-Free_to_use-green)

---

## Quick Start

### Prerequisites

- **Node.js 14+** — [download](https://nodejs.org/)
- **Git** — [download](https://git-scm.com/)

### 1. Install dependencies

```bash
cd os-portfolio
npm install
```

### 2. Run locally

```bash
npm start
```

Opens at [http://localhost:3000](http://localhost:3000). Changes auto-reload.

### 3. Build for production

```bash
npm run build
```

---

## Deploy to GitHub Pages (Free)

### First time

1. **Create a GitHub repo** named `yourusername.github.io` (public, no README).

2. **Edit `package.json`** — set `"homepage"` to `"https://yourusername.github.io"`.

3. **Push your code:**

   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/yourusername/yourusername.github.io.git
   git push -u origin main
   ```

4. **Deploy:**

   ```bash
   npm run deploy
   ```

5. **Enable Pages** — go to repo Settings → Pages → Source: `gh-pages` branch → Save.

Your site goes live at `https://yourusername.github.io` within a few minutes.

### Updating

```bash
git add .
git commit -m "Your changes"
git push origin main
npm run deploy
```

---

## Customisation

All personal content lives in **`src/App.jsx`** at the top of the file in clearly labelled constants:

| What                | Where to edit                |
| ------------------- | ---------------------------- |
| Name / title        | `PROFILE` object             |
| About cards         | `ABOUT_CARDS` array          |
| Skills              | `SKILLS` array               |
| Projects            | `PROJECTS` array             |
| Contact links       | `CONTACT_LINKS` array        |
| Terminal files       | `FILE_SYSTEM` object         |
| Colours             | `src/App.css` `:root` block  |
| Background gradient | `src/App.css` `.os` rule     |
| Page title          | `public/index.html` `<title>`|

### Add a new dock app

1. Create a new function component (e.g. `SkillsPanel`).
2. Add an entry to the `APPS` array:
   ```js
   { name: 'Skills', icon: '🎯', Component: SkillsPanel },
   ```

### Add a terminal command

Add a key to the object returned by `buildCommands`:

```js
skills: () => 'React, Node, Python, …',
```

---

## Troubleshooting

| Problem                     | Fix                                                            |
| --------------------------- | -------------------------------------------------------------- |
| `npm install` fails         | Delete `node_modules` and `package-lock.json`, then retry.     |
| Port 3000 busy              | `PORT=3001 npm start`                                          |
| GitHub Pages 404            | Check repo name matches username; wait 5–10 min after deploy.  |
| Changes not showing         | Hard refresh (`Ctrl+Shift+R` / `Cmd+Shift+R`).                |

---

## License

Free to use for personal portfolios. Attribution appreciated but not required.
