---
title: "CSS Grid Basics"
date: 2025-01-05
topic: css
---

Quick notes on CSS Grid fundamentals.

## Container Properties

```css
.container {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  grid-template-rows: auto;
  gap: 1rem;
}
```

## Key Concepts

- `fr` unit - fraction of available space
- `repeat()` - shorthand for repeating patterns
- `gap` - spacing between grid items
- `grid-template-areas` - named areas for layout

## Useful Patterns

**Auto-fit responsive grid:**
```css
grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
```

This creates a responsive grid without media queries!
