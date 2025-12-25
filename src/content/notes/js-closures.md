---
title: "JavaScript Closures Explained"
date: 2025-01-10
topic: javascript
---

A closure is when a function "remembers" variables from its outer scope even after the outer function has returned.

## Simple Example

```javascript
function createCounter() {
  let count = 0;
  return function() {
    count++;
    return count;
  };
}

const counter = createCounter();
counter(); // 1
counter(); // 2
counter(); // 3
```

The inner function "closes over" the `count` variable.

## Common Use Cases

1. **Data privacy** - Create private variables
2. **Function factories** - Generate specialized functions
3. **Callbacks** - Preserve state in async operations
4. **Memoization** - Cache expensive computations

## Gotcha: Loops

```javascript
// Problem
for (var i = 0; i < 3; i++) {
  setTimeout(() => console.log(i), 100); // 3, 3, 3
}

// Solution: use let or IIFE
for (let i = 0; i < 3; i++) {
  setTimeout(() => console.log(i), 100); // 0, 1, 2
}
```
