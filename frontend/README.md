# LangScope Frontend

A hand-drawn, sketch-style React frontend for the LangScope LLM evaluation platform.

## Features

- **Sketch UI**: Hand-drawn aesthetic using [Rough.js](https://roughjs.com/) for borders, buttons, and charts
- **Grid Paper Background**: Graph paper styling throughout the app
- **Sticky Notes**: Yellow sticky notes for alerts, leaderboards, and notifications
- **10-Dimensional Ratings**: Visualize model performance across multiple dimensions
- **Arena Mode**: Interactive model comparison with user feedback
- **Domain Leaderboards**: Rankings by domain (code, medical, legal, etc.)

## Tech Stack

- **React 18** + TypeScript
- **Vite** for fast development
- **Rough.js** for hand-drawn graphics
- **React Router** for navigation
- **TanStack Query** for data fetching
- **Zustand** for state management
- **CSS Modules** for styling

## Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Copy environment file
cp .env.example .env

# Start development server
npm run dev
```

The app will be available at `http://localhost:3000`.

### Environment Variables

```bash
VITE_API_BASE_URL=http://localhost:8000
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your-anon-key
```

## Project Structure

```
src/
├── api/                  # API client and React Query hooks
│   ├── client.ts         # Fetch wrapper with auth
│   ├── hooks/            # useModels, useLeaderboard, etc.
│   └── types/            # TypeScript interfaces
├── components/
│   ├── sketch/           # Rough.js components (Button, Card, Input)
│   ├── sticky/           # Sticky note components
│   ├── charts/           # Hand-drawn chart components
│   └── layout/           # GridPaper, Sidebar, PageLayout
├── pages/
│   ├── Home/             # Landing page with search
│   ├── Dashboard/        # Admin overview with charts
│   ├── Rankings/         # Leaderboard table
│   ├── Arena/            # Interactive model battles
│   └── About/            # Project information
├── styles/
│   ├── globals.css       # Grid paper, fonts, CSS variables
│   └── sketch-theme.ts   # Color palette and Rough.js config
└── App.tsx               # Routes and providers
```

## Available Scripts

```bash
npm run dev       # Start development server
npm run build     # Build for production
npm run preview   # Preview production build
npm run lint      # Run ESLint
npm run test      # Run unit tests
npm run test:e2e  # Run Playwright E2E tests
```

## Design System

### Colors

| Variable | Color | Usage |
|----------|-------|-------|
| `--color-paper` | `#fafaf8` | Background |
| `--color-grid` | `#e8e8e0` | Grid lines |
| `--color-ink` | `#2d2d2d` | Primary text |
| `--color-pencil` | `#666666` | Secondary text |
| `--color-sticky-yellow` | `#fff740` | Sticky notes |
| `--color-accent` | `#4a90d9` | Links, highlights |

### Fonts

- **Patrick Hand**: Body text, handwriting style
- **Architects Daughter**: Headings, titles
- **Caveat**: Accents, numbers

### Components

- `<SketchButton>` - Hand-drawn button with Rough.js border
- `<SketchCard>` - Card with sketchy border
- `<SketchInput>` - Input field with hand-drawn styling
- `<StickyNote>` - Yellow sticky note with pin
- `<LeaderboardSticky>` - Mini leaderboard on a sticky

## API Integration

The frontend connects to the LangScope FastAPI backend. Key endpoints:

- `GET /models` - List all models
- `GET /domains` - List all domains
- `GET /leaderboard/{domain}` - Get rankings
- `POST /arena/sessions` - Start arena session
- `GET /recommendations` - Get model recommendations

## License

MIT
