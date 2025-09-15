# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Structure

This is a modern personal portfolio website built with React, Vite, and Tailwind CSS. The main application code is located in the `website/` directory, which contains a complete React application following modern best practices.

**Key directories:**
- `website/` - Main React application
- `website/src/components/` - React components (ModernHeader, Footer, HomePage, AboutMe, Resume, etc.)
- `website/src/components/ui/` - Reusable UI components (Button, Card, Badge, SkipLink, etc.)
- `website/src/hooks/` - Custom React hooks (useReducedMotion)
- `website/public/` - Static assets including resume.pdf

## Development Commands

All development commands should be run from the `website/` directory:

```bash
cd website
```

**Development server:**
```bash
pnpm dev
```

**Build for production:**
```bash
pnpm build
```

**Lint code:**
```bash
pnpm lint
```

**Preview production build:**
```bash
pnpm preview
```

## Modern Architecture

**Frontend Framework:** React 18 with Vite as the build tool
**Styling:** Tailwind CSS v3 with custom Miami-themed design tokens
**Animations:** Framer Motion for microinteractions and page transitions
**Icons:** Lucide React for consistent iconography
**SEO:** React Helmet Async for meta tag management
**Routing:** React Router DOM v7 for client-side navigation

**Design System:**
- Miami University color palette (greens and oranges)
- Consistent spacing and typography scales
- Accessible focus states and skip links
- Dark mode support with proper contrast ratios
- Responsive design with mobile-first approach

**Key Components:**
- `ModernHeader.jsx` - Responsive navigation with mobile menu
- `Resume.jsx` - ATS-friendly HTML resume with PDF download tracking
- `ModernProjectsPage.jsx` - Enhanced projects showcase with animations
- `SEO.jsx` - Meta tags and structured data management
- `ui/` components - Reusable design system components

**Performance Features:**
- Optimized images with loading states
- Code splitting and lazy loading
- CSS purging with Tailwind
- Proper meta tags for SEO
- Analytics event tracking for resume downloads

## Package Management

This project uses `pnpm` as the package manager. Key dependencies include:
- React 18 with modern hooks
- Tailwind CSS for styling
- Framer Motion for animations
- Lucide React for icons
- React Helmet Async for SEO

## Development Notes

**Configuration Files:**
- `tailwind.config.cjs` - Tailwind CSS configuration with Miami theme
- `postcss.config.cjs` - PostCSS configuration
- Files use `.cjs` extension due to ES modules in package.json

**Accessibility:**
- Skip links for keyboard navigation
- Proper semantic HTML structure
- ARIA labels and screen reader support
- Focus management and reduced motion support

**SEO Features:**
- Structured data with schema.org Person markup
- Open Graph and Twitter Card meta tags
- Canonical URLs and proper page titles
- Optimized descriptions for each page

## Resume Integration

The site includes a comprehensive resume page (`/resume`) that:
- Displays HTML version of resume for ATS compatibility
- Includes PDF download with analytics tracking
- Features proper print styles for physical printing
- Contains structured data for search engines

## Deployment

The project includes `vercel.json` configuration for Vercel deployment with proper routing for SPA.