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
const site = {
  name: "Tang's Machine Learning Blog",
  eyebrow: "Research notes on machine learning, optimization, and reliable AI systems",
  headline: "Tang's Machine Learning Blog",
  description:
    "A LaTeX-native research blog for machine learning notes, mathematical derivations, reproducible experiments, and carefully typeset code.",
  repo: "https://github.com/TXJ2006/elegantbook-tech-blog",
};

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
    lang: meta.lang ?? "en",
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
<html lang="${htmlEscape(post.lang)}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${htmlEscape(post.title)} | ${htmlEscape(site.name)}</title>
  <link rel="stylesheet" href="../../site.css">
</head>
<body class="reader">
  <header class="topbar">
    <a class="brand" href="../../">${htmlEscape(site.name)}</a>
    <nav>
      <a href="${site.repo}">GitHub</a>
      <a href="../../pdf/${encodeURIComponent(post.slug)}.pdf">PDF</a>
    </nav>
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
        <a href="posts/${encodeURIComponent(post.slug)}/">Read online</a>
        <a href="pdf/${encodeURIComponent(post.slug)}.pdf">Open PDF</a>
      </div>
    </article>`).join("\n");

  writeFileSync(path.join(publicDir, "index.html"), `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${htmlEscape(site.name)}</title>
  <link rel="stylesheet" href="site.css">
</head>
<body>
  <header class="hero">
    <nav>
      <strong>${htmlEscape(site.name)}</strong>
      <a href="${site.repo}">GitHub</a>
    </nav>
    <div>
      <p>${htmlEscape(site.eyebrow)}</p>
      <h1>${htmlEscape(site.headline)}</h1>
      <span>${htmlEscape(site.description)}</span>
    </div>
  </header>
  <main class="content">
    <section class="post-list" aria-label="Published notes">
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
  --ink: #171512;
  --muted: #6a6259;
  --line: #ded6cc;
  --paper: #fbfaf7;
  --accent: #7a3d25;
  --accent-dark: #4f281a;
  --wash: #f3efe8;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Times New Roman", Times, serif;
  color: var(--ink);
  background: var(--paper);
}
a { color: inherit; text-decoration: none; }
.hero {
  min-height: 58vh;
  padding: 28px clamp(20px, 5vw, 72px) 56px;
  background:
    linear-gradient(135deg, rgba(39, 31, 25, 0.84), rgba(122, 61, 37, 0.68)),
    url("https://images.unsplash.com/photo-1457369804613-52c61a468e7d?auto=format&fit=crop&w=1800&q=80");
  background-size: cover;
  background-position: center;
  color: #fffaf0;
  display: grid;
  grid-template-rows: auto 1fr;
}
.hero nav, .topbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 18px;
}
.hero nav strong,
.brand {
  font-family: Georgia, "Times New Roman", Times, serif;
  font-style: italic;
  font-weight: 500;
}
.hero nav a, .topbar a { border-bottom: 1px solid currentColor; }
.topbar nav {
  display: flex;
  gap: 18px;
}
.hero div {
  align-self: end;
  max-width: 980px;
}
.hero p {
  margin: 0 0 16px;
  max-width: 760px;
  font-size: 1rem;
  line-height: 1.5;
}
.hero h1 {
  margin: 0 0 20px;
  max-width: 1080px;
  font-family: Georgia, "Times New Roman", Times, serif;
  font-style: italic;
  font-weight: 500;
  font-size: clamp(3rem, 7.8vw, 8.2rem);
  line-height: 0.93;
  letter-spacing: 0;
}
.hero span {
  display: block;
  max-width: 820px;
  font-size: clamp(1.06rem, 2vw, 1.3rem);
  line-height: 1.65;
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
  background: #fffdf8;
  padding: 22px;
}
.post-card time {
  color: var(--muted);
  font-size: 0.92rem;
}
.post-card h2 {
  margin: 10px 0 12px;
  font-family: Georgia, "Times New Roman", Times, serif;
  font-style: italic;
  font-weight: 500;
  font-size: 1.55rem;
  line-height: 1.22;
}
.post-card p {
  min-height: 3.2em;
  color: var(--muted);
  line-height: 1.65;
}
.actions {
  display: flex;
  gap: 14px;
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
  background: rgba(251, 250, 247, 0.94);
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
  font-family: Georgia, "Times New Roman", Times, serif;
  font-style: italic;
  font-weight: 500;
  font-size: clamp(2.2rem, 5vw, 4.6rem);
  line-height: 1.02;
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
  .hero h1 { font-size: clamp(2.7rem, 16vw, 4.8rem); }
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
