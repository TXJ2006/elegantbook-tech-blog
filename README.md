# ElegantBook LaTeX 技术博客

这是一个可以直接写 LaTeX、自动编译并发布到 GitHub Pages 的技术博客骨架。每篇文章放在 `posts/*.tex`，构建脚本会生成：

- `public/index.html`：公开博客首页
- `public/posts/<slug>/index.html`：每篇文章的在线 PDF 预览页
- `public/pdf/<slug>.pdf`：ElegantBook 排版后的 PDF

## 本地写作

1. 复制示例文章：

   ```powershell
   Copy-Item posts/hello-elegantbook.tex posts/my-first-post.tex
   ```

2. 修改文件顶部的元信息：

   ```tex
   % blog-title: 文章标题
   % blog-date: 2026-06-15
   % blog-description: 首页展示的摘要
   ```

3. 编译并生成网站：

   ```powershell
   node scripts/build-site.mjs
   ```

4. 打开 `public/index.html` 预览。

## GitHub Pages 部署

1. 在 GitHub 新建一个仓库。
2. 将本目录推送到仓库：

   ```powershell
   git init
   git add .
   git commit -m "Initial ElegantBook LaTeX blog"
   git branch -M main
   git remote add origin https://github.com/<your-name>/<your-repo>.git
   git push -u origin main
   ```

3. 在 GitHub 仓库进入 `Settings -> Pages`，将 `Build and deployment` 的 Source 设为 `GitHub Actions`。
4. 之后每次 push，`.github/workflows/deploy.yml` 会自动编译 LaTeX 并发布。

## 写作约定

- 文章必须是独立的 `.tex` 文件，放在 `posts/` 下。
- 使用 `\input{tex/blog-preamble}` 加载中文字体、链接样式和代码高亮。
- 推荐使用 `codeblock` 环境写代码：

  ```tex
  \begin{codeblock}{Python}
  print("hello")
  \end{codeblock}
  ```

## 与 Overleaf 接轨

本项目已经内置 `elegantbook.cls`，来自本机 MiKTeX 安装的 ElegantBook 4.7。你也可以把整个目录上传到 Overleaf，入口文件选择 `posts/hello-elegantbook.tex`，编译器选择 XeLaTeX。
