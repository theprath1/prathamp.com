# Pratham Patel - Portfolio

A personal portfolio and blog built with Astro and Tailwind CSS, showcasing projects, blog posts, and notes.

## Features

- **Projects** - Showcase of data science and AI projects
- **Blog** - Technical articles and thoughts
- **Notes** - Quick learning notes and references
- **Dark Mode** - Theme toggle for light/dark modes
- **Responsive** - Mobile-friendly design
- **Content Collections** - Markdown-based content management

## Tech Stack

- [Astro](https://astro.build/) - Static site generator
- [Tailwind CSS](https://tailwindcss.com/) - Utility-first CSS framework
- TypeScript

## Project Structure

```
/
├── public/              # Static assets
├── src/
│   ├── components/      # Reusable UI components
│   ├── config/          # Site configuration
│   ├── content/         # Markdown content (projects, blog, notes)
│   ├── layouts/         # Page layouts
│   ├── pages/           # Route pages
│   ├── styles/          # Global styles
│   └── utils/           # Utility functions
└── package.json
```

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
# Clone the repository
git clone https://github.com/theprathpatel/my-portfolio.git

# Navigate to the project
cd my-portfolio

# Install dependencies
npm install
```

### Development

```bash
# Start development server
npm run dev
```

Open [http://localhost:4321](http://localhost:4321) in your browser.

### Build

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

## Adding Content

### Projects

Add new projects in `src/content/projects/` as Markdown files with frontmatter:

```markdown
---
title: "Project Title"
description: "Brief description"
tech: ["Python", "TensorFlow"]
featured: true
draft: false
---

Project content here...
```

### Blog Posts

Add blog posts in `src/content/blog/` with frontmatter:

```markdown
---
title: "Post Title"
description: "Brief description"
date: 2024-01-01
tags: ["tag1", "tag2"]
draft: false
---

Post content here...
```

## License

MIT
