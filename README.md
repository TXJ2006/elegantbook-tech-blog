# Tang's Machine Learning Blog

This is a LaTeX-native machine learning blog based on ElegantBook. Write posts as standalone `.tex` files, push to GitHub, and GitHub Actions will compile and publish the site to GitHub Pages.

The build generates:

- `public/index.html`: the public blog homepage
- `public/posts/<slug>/index.html`: the online PDF reader page for each post
- `public/pdf/<slug>.pdf`: the ElegantBook PDF for each post

## Writing

Copy one of the templates:

```powershell
Copy-Item templates/post-en.tex posts/my-first-note.tex
Copy-Item templates/post-cn.tex posts/my-first-cn-note.tex
```

Update the post metadata:

```tex
% blog-title: Your Post Title
% blog-date: 2026-06-15
% blog-description: A short abstract for the homepage.
% blog-lang: en
```

Use English posts with:

```tex
\documentclass[en,nofont,11pt]{elegantbook}
```

Use Chinese posts with:

```tex
\documentclass[cn,nofont,11pt]{elegantbook}
```

Both templates load `tex/blog-preamble.tex`, which provides CJK font fallback, code highlighting, and shared link styling.

## Local Build

```powershell
node scripts/build-site.mjs
```

Open `public/index.html` to preview the generated site.

## Deployment

The repository is configured for GitHub Pages with GitHub Actions. Every push to `main` compiles all posts in `posts/` and deploys the generated site.

## Code Highlighting

Use the `codeblock` environment:

```tex
\begin{codeblock}{Python}
print("hello")
\end{codeblock}
```

## Overleaf

This project includes `elegantbook.cls` from ElegantBook 4.7. You can upload the repository to Overleaf, choose a post under `posts/` as the main file, and compile with XeLaTeX.
