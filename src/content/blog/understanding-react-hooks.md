---
title: "Understanding React Hooks"
description: "A deep dive into useState and useEffect - the two most commonly used React hooks."
date: 2025-01-15
tags: [react, javascript, tutorial]
---

React Hooks revolutionized how we write React components. Let's explore the two most important ones.

## useState

The `useState` hook lets you add state to functional components:

```jsx
const [count, setCount] = useState(0);
```

### Key Points

1. Returns an array with two elements: current state and setter function
2. Can hold any type of value (primitives, objects, arrays)
3. Setter can take a value or a function for updates based on previous state

## useEffect

The `useEffect` hook handles side effects in your components:

```jsx
useEffect(() => {
  document.title = `Count: ${count}`;
}, [count]);
```

### When to Use

- Fetching data
- Setting up subscriptions
- Manually changing the DOM
- Logging

## Best Practices

- Keep effects focused on a single concern
- Always specify dependencies
- Clean up subscriptions and timers

Happy coding!
