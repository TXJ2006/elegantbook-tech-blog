import { execFileSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const postsDir = path.join(root, "posts");
const buildDir = path.join(root, "build");
const publicDir = path.join(root, "public");
const pdfDir = path.join(publicDir, "pdf");
const postPagesDir = path.join(publicDir, "posts");

function ensureDir(dir) {
  mkdirSync(dir, { recursive: true });
}

function htmlEscape(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function readMeta(texPath) {
  const source = readFileSync(texPath, "utf8");
  const meta = {};
  for (const line of source.split(/\r?\n/)) {
    const match = line.match(/^%\s*blog-([a-z-]+):\s*(.+?)\s*$/i);
    if (match) meta[match[1]] = match[2];
  }
  const slug = path.basename(texPath, ".tex");
  return {
    slug,
    title: meta.title ?? slug,
    date: meta.date ?? "",
    description: meta.description ?? "",
  };
}

function compilePost(texPath, slug) {
  const outDir = path.join(buildDir, "tex", slug);
  ensureDir(outDir);
  const env = {
    ...process.env,
    TEXINPUTS: `.${path.delimiter}${root}${path.delimiter}${path.join(root, "tex")}${path.delimiter}`,
  };
  try {
    execFileSync("latexmk", [
      "-xelatex",
      "-interaction=nonstopmode",
      "-halt-on-error",
      `-outdir=${outDir}`,
      texPath,
    ], { cwd: root, stdio: "inherit", env });
  } catch {
    console.warn("latexmk failed; falling back to two xelatex passes.");
    for (let pass = 0; pass < 2; pass += 1) {
      execFileSync("xelatex", [
        "-interaction=nonstopmode",
        "-halt-on-error",
        `-output-directory=${outDir}`,
        texPath,
      ], { cwd: root, stdio: "inherit", env });
    }
  }
  const pdfPath = path.join(outDir, `${slug}.pdf`);
  if (!existsSync(pdfPath)) throw new Error(`Missing compiled PDF: ${pdfPath}`);
  copyFileSync(pdfPath, path.join(pdfDir, `${slug}.pdf`));
}

function renderPostPage(post) {
  const dir = path.join(postPagesDir, post.slug);
  ensureDir(dir);
  writeFileSync(path.join(dir, "index.html"), `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${htmlEscape(post.title)}</title>
  <link rel="stylesheet" href="../../site.css">
</head>
<body class="reader">
  <header class="topbar">
    <a href="../../">Tech Notes</a>
    <a href="../../pdf/${encodeURIComponent(post.slug)}.pdf">下载 PDF</a>
  </header>
  <main>
    <section class="post-head">
      <p>${htmlEscape(post.date)}</p>
      <h1>${htmlEscape(post.title)}</h1>
      <span>${htmlEscape(post.description)}</span>
    </section>
    <iframe class="pdf-frame" src="../../pdf/${encodeURIComponent(post.slug)}.pdf" title="${htmlEscape(post.title)}"></iframe>
  </main>
</body>
</html>
`, "utf8");
}

function renderIndex(posts) {
  const items = posts.map((post) => `
    <article class="post-card">
      <time>${htmlEscape(post.date)}</time>
      <h2><a href="posts/${encodeURIComponent(post.slug)}/">${htmlEscape(post.title)}</a></h2>
      <p>${htmlEscape(post.description)}</p>
      <div class="actions">
        <a href="posts/${encodeURIComponent(post.slug)}/">在线阅读</a>
        <a href="pdf/${encodeURIComponent(post.slug)}.pdf">PDF</a>
      </div>
    </article>`).join("\n");

  writeFileSync(path.join(publicDir, "index.html"), `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tech Notes</title>
  <link rel="stylesheet" href="site.css">
</head>
<body>
  <header class="hero">
    <nav>
      <strong>Tech Notes</strong>
      <a href="https://github.com/">GitHub</a>
    </nav>
    <div>
      <p>LaTeX-first technical writing</p>
      <h1>用 ElegantBook 写公开技术博客</h1>
      <span>每篇文章用 LaTeX 编写，GitHub Actions 自动编译成 PDF，并发布到 GitHub Pages。</span>
    </div>
  </header>
  <main class="content">
    <section class="post-list">
      ${items}
    </section>
  </main>
</body>
</html>
`, "utf8");
}

function renderCss() {
  writeFileSync(path.join(publicDir, "site.css"), `
:root {
  color-scheme: light;
  --ink: #172026;
  --muted: #65717a;
  --line: #d9e2e7;
  --paper: #fbfcfb;
  --accent: #1f6f8b;
  --accent-dark: #14495d;
  --wash: #eef5f3;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background: var(--paper);
}
a { color: inherit; text-decoration: none; }
.hero {
  min-height: 58vh;
  padding: 28px clamp(20px, 5vw, 72px) 56px;
  background:
    linear-gradient(140deg, rgba(31,111,139,0.92), rgba(23,32,38,0.88)),
    url("https://images.unsplash.com/photo-1457369804613-52c61a468e7d?auto=format&fit=crop&w=1800&q=80");
  background-size: cover;
  background-position: center;
  color: white;
  display: grid;
  grid-template-rows: auto 1fr;
}
.hero nav, .topbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 18px;
}
.hero nav a, .topbar a { border-bottom: 1px solid currentColor; }
.hero div {
  align-self: end;
  max-width: 860px;
}
.hero p {
  margin: 0 0 12px;
  font-size: 0.86rem;
  letter-spacing: 0;
  text-transform: uppercase;
}
.hero h1 {
  margin: 0 0 18px;
  font-size: clamp(2.3rem, 6vw, 5.8rem);
  line-height: 1;
  letter-spacing: 0;
}
.hero span {
  display: block;
  max-width: 760px;
  font-size: clamp(1rem, 2vw, 1.24rem);
  line-height: 1.7;
}
.content {
  width: min(1080px, calc(100% - 40px));
  margin: 42px auto 80px;
}
.post-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 18px;
}
.post-card {
  border: 1px solid var(--line);
  border-radius: 8px;
  background: white;
  padding: 22px;
}
.post-card time {
  color: var(--muted);
  font-size: 0.88rem;
}
.post-card h2 {
  margin: 10px 0 12px;
  font-size: 1.35rem;
  line-height: 1.25;
}
.post-card p {
  min-height: 3.2em;
  color: var(--muted);
  line-height: 1.65;
}
.actions {
  display: flex;
  gap: 12px;
  margin-top: 18px;
}
.actions a {
  color: var(--accent-dark);
  font-weight: 700;
}
.topbar {
  position: sticky;
  top: 0;
  z-index: 2;
  padding: 14px clamp(16px, 4vw, 48px);
  background: rgba(251,252,251,0.94);
  border-bottom: 1px solid var(--line);
  backdrop-filter: blur(12px);
}
.reader main {
  width: min(1180px, calc(100% - 28px));
  margin: 26px auto 48px;
}
.post-head {
  margin: 0 0 18px;
}
.post-head p {
  margin: 0 0 8px;
  color: var(--muted);
}
.post-head h1 {
  margin: 0 0 10px;
  font-size: clamp(1.9rem, 4vw, 3.4rem);
  line-height: 1.08;
}
.post-head span {
  color: var(--muted);
  line-height: 1.7;
}
.pdf-frame {
  display: block;
  width: 100%;
  height: 78vh;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--wash);
}
@media (max-width: 720px) {
  .hero { min-height: 62vh; }
  .pdf-frame { height: 70vh; }
}
`, "utf8");
}

function main() {
  rmSync(buildDir, { recursive: true, force: true });
  rmSync(publicDir, { recursive: true, force: true });
  ensureDir(pdfDir);
  ensureDir(postPagesDir);

  const posts = readdirSync(postsDir)
    .filter((name) => name.endsWith(".tex"))
    .map((name) => path.join(postsDir, name))
    .map((texPath) => {
      const post = readMeta(texPath);
      compilePost(texPath, post.slug);
      renderPostPage(post);
      return post;
    })
    .sort((a, b) => b.date.localeCompare(a.date));

  renderCss();
  writeFileSync(path.join(publicDir, ".nojekyll"), "", "utf8");
  renderIndex(posts);
  console.log(`Built ${posts.length} post(s) into ${publicDir}`);
}

main();
