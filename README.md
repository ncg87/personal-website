# Personal Portfolio Website

A modern, high-performance personal portfolio website built with React and Tailwind CSS. Features terminal-style animations, comprehensive project showcases, blog functionality, and optimized performance with code splitting and lazy loading.

## 🚀 Features

### Core Functionality
- **Terminal-Style Homepage**: Interactive boot sequence with skip animation option
- **Project Showcase**: Detailed case studies with live demos and GitHub links
- **Blog System**: Featured posts, individual article pages, and content management
- **Professional Resume**: ATS-friendly HTML version with PDF download tracking
- **Contact Form**: Backend integration with multiple service support and fallback
- **404 Error Page**: Branded error handling with helpful navigation

### User Experience
- **Dark/Light Mode**: Complete theme system with system preference detection
- **Responsive Design**: Mobile-first approach with touch-friendly interactions
- **Accessibility**: WCAG 2.2 AA compliance with screen reader support
- **Performance**: Code splitting, lazy loading, and optimized bundle sizes
- **SEO Optimized**: Meta tags, structured data, and Open Graph integration
- **Analytics**: Google Analytics integration with event tracking

### Technical Excellence
- **Progressive Web App**: Service worker and offline functionality ready
- **Bundle Optimization**: Split from 514KB to ~141KB largest chunk (45KB gzipped)
- **Animation System**: Respects reduced motion preferences
- **Loading States**: Skeleton screens and smooth transitions

## 🛠 Technologies Used

### Frontend Framework
- **React 18**: Modern hooks, Suspense, and concurrent features
- **Vite**: Lightning-fast development and optimized production builds
- **React Router DOM v7**: Client-side routing with lazy loading

### Styling & UI
- **Tailwind CSS v3**: Utility-first CSS framework with custom Miami theme
- **Framer Motion**: Advanced animations and microinteractions
- **Lucide React**: Consistent iconography system
- **Custom Design System**: Reusable UI components (Button, Card, Badge, etc.)

### Performance & SEO
- **React Helmet Async**: Dynamic meta tag management
- **Code Splitting**: Manual chunks for optimal loading
- **Image Optimization**: Lazy loading and responsive images
- **Bundle Analyzer**: Webpack bundle optimization

### Backend Integration
- **Google Analytics**: Event tracking and page view analytics
- **Formspree/Netlify Forms**: Contact form backend options
- **Environment Configuration**: Secure API key management

### Development Tools
- **ESLint**: Code quality and consistency
- **PostCSS**: CSS processing and optimization
- **Git Hooks**: Pre-commit quality checks

## 📁 Project Structure

```
website/
├── src/
│   ├── components/           # React components
│   │   ├── ui/              # Reusable UI components
│   │   ├── ModernHeader.jsx # Navigation with theme toggle
│   │   ├── TerminalHomePage.jsx # Interactive homepage
│   │   ├── ModernProjectsPage.jsx # Project showcase
│   │   ├── BlogPage.jsx     # Blog listing
│   │   └── ...
│   ├── contexts/            # React contexts (Theme, etc.)
│   ├── hooks/              # Custom React hooks
│   ├── utils/              # Utility functions (analytics, etc.)
│   ├── data/               # Static data (projects, etc.)
│   └── styles/             # Global styles and Tailwind config
├── public/                 # Static assets
└── dist/                  # Production build
```

## 🎨 Design System

### Color Palette
- **Primary**: Miami Green (#005030) - University of Miami brand
- **Secondary**: Miami Orange (#f97316) - Complementary accent
- **Neutral**: Sophisticated grayscale with dark mode variants
- **Semantic**: Success, warning, error states

### Typography
- **Headings**: Bold, hierarchical scaling
- **Body**: Optimized for readability across devices
- **Code**: Monospace fonts for terminal aesthetics

### Components
- **Cards**: Consistent padding and hover effects
- **Buttons**: Primary, secondary, outline, and ghost variants
- **Badges**: Color-coded project categories and tags
- **Forms**: Accessible with validation states

## 📱 Pages

1. **Home (`/`)**: Terminal-style animated introduction with project highlights
2. **About (`/about`)**: Professional background, education, and experience timeline
3. **Projects (`/projects`)**: Comprehensive project portfolio with filtering
4. **Blog (`/posts`)**: Technical articles and project deep-dives
5. **Resume (`/resume`)**: Professional HTML resume with PDF download
6. **Contact (`/contact`)**: Multi-backend contact form with validation

## ⚡ Performance Metrics

- **Bundle Size**: Optimized from 514KB to multiple chunks (largest 141KB)
- **Loading**: Initial page load < 2s, subsequent navigation < 500ms
- **SEO Score**: 100/100 (Google PageSpeed Insights)
- **Accessibility**: WCAG 2.2 AA compliant
- **Mobile Performance**: 95+ (Google PageSpeed Insights)

