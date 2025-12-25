---
title: "Task Manager App"
description: "A full-stack task management application with real-time updates, user authentication, and a clean, intuitive interface."
tech: [React, Node.js, PostgreSQL, Socket.io]
github: "https://github.com/pratham/task-manager"
featured: true
---

## Overview

A productivity tool I built to learn full-stack development. It handles everything from user auth to real-time collaborative features.

## Features

- **User Authentication** - Secure login with JWT
- **Real-time Updates** - See changes instantly with WebSockets
- **Drag & Drop** - Intuitive task organization
- **Labels & Filters** - Organize tasks your way
- **Due Dates** - Never miss a deadline

## Tech Stack

### Frontend
- React with hooks for state management
- Tailwind CSS for styling
- React Beautiful DnD for drag-and-drop

### Backend
- Node.js with Express
- PostgreSQL for data persistence
- Socket.io for real-time communication

## Architecture

The app follows a clean separation of concerns:

```
client/
  src/
    components/
    hooks/
    services/
server/
  routes/
  controllers/
  models/
```

## What I Learned

- WebSocket integration patterns
- Database design for collaborative apps
- State management in complex React apps
