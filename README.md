# Drawing & Gallery Application

A web-based drawing application that combines a custom drawing language with AI-powered generation and a social gallery system.

## Features

### 🎨 Drawing Board

- **Dual Input Modes**
  - Traditional command-based drawing
  - AI-powered natural language drawing generation
- **Real-time Preview**
- **Save Functionality** (authenticated users)

### 🖼️ Gallery System

- Browse community artwork
- Like/Unlike system
- Sequential navigation
- Social interaction features

### 🔐 Authentication

- Protected features for registered users
- Seamless sign-in integration
- Personalized experience

## Drawing Language Reference

### LINE Command

LINE startX startY endX endY color

Draws a straight line between two points.

**Example:**

LINE 0 0 2 0 black   # Horizontal black line

### ARC Command

ARC centerX centerY radius startAngle endAngle  clockwise rotation color

Creates curved arcs as part of a circle.

**Example:**

ARC 1 1 0.8 1 1.2 1 180 0 true 0 blue # Blue semicircle

## Usage Guide

### Drawing Board Mode

1. **Normal Mode**
   - Enter drawing commands in the textarea
   - Click "Draw" to execute
   - Save your work (requires authentication)

2. **AI Mode** (Authenticated Users)
   - Toggle AI mode with the switch button
   - Describe your desired drawing in natural language
   - AI converts descriptions to drawing commands

### Gallery Mode

- Navigate through artwork using Previous/Next buttons
- Like/Unlike artwork (authenticated users)
- View popularity through like counts



