import React, { useEffect, Suspense } from 'react';
import { BrowserRouter as Router, Route, Routes, useLocation } from 'react-router-dom';
import { trackPageView } from './utils/analytics';
import { ThemeProvider } from './contexts/ThemeContext';
import ModernHeader from './components/ModernHeader';
import Footer from './components/Footer';
import TerminalHomePage from './components/TerminalHomePage';
import Box from '@mui/material/Box';
import SkipLink from './components/ui/SkipLink';
import { PageLoader } from './components/ui/LoadingSpinner';

// Lazy load components for code splitting
const ModernAboutPage = React.lazy(() => import('./components/ModernAboutPage'));
const Resume = React.lazy(() => import('./components/Resume'));
const ContactForm = React.lazy(() => import('./components/ContactForm'));
const ModernProjectsPage = React.lazy(() => import('./components/ModernProjectsPage'));
const ProjectCaseStudy = React.lazy(() => import('./components/ProjectCaseStudy'));
const NotFoundPage = React.lazy(() => import('./components/NotFoundPage'));
const BlogPage = React.lazy(() => import('./components/BlogPage'));
const BlogPost = React.lazy(() => import('./components/BlogPost'));
const PlaceholderPage = React.lazy(() => import('./components/PlaceHolder'));

const AppContent = () => {
    const location = useLocation();
    
    // Track page views
    useEffect(() => {
        trackPageView(location.pathname, document.title);
    }, [location]);
    
    return (
        <>
            <SkipLink />
            {/* App Container */}
            <Box
                sx={{
                    backgroundColor: 'rgba(0, 80, 48, 0.3)', // Background overlay color
                    position: 'relative',
                    minHeight: '100vh', // Ensures the app spans the entire viewport
                    width: '100vw',
                    overflow: 'hidden', // Prevent content overflow
                }}
            >
                {/* Background Image */}
                <Box
                    sx={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        backgroundImage: 'url(/campus1.jpg)', // Path to your background image
                        backgroundSize: 'cover',
                        backgroundRepeat: 'no-repeat',
                        backgroundPosition: 'center',
                        zIndex: -1, // Ensures the background is behind all content
                    }}
                />

                {/* Content Wrapper */}
                <Box
                    sx={{
                        position: 'relative',
                        display: 'flex',
                        flexDirection: 'column',
                        minHeight: '100vh', // Ensures the wrapper spans the viewport height
                    }}
                >
                    <ModernHeader />
                    <Box
                        component="main"
                        id="main-content"
                        sx={{
                            flex: 1, // Makes the main content stretch between the header and footer
                        }}
                    >
                        <Suspense fallback={<PageLoader />}>
                            <Routes location={location} key={location.pathname}>
                                <Route path="/" element={<TerminalHomePage key="homepage" />} />
                                <Route path="/about" element={<ModernAboutPage />} />
                                <Route path="/resume" element={<Resume />} />
                                <Route path="/contact" element={<ContactForm />} />
                                <Route path="/posts" element={<BlogPage />} />
                                <Route path="/projects" element={<ModernProjectsPage />} />
                                <Route path="/projects/:slug" element={<ProjectCaseStudy />} />
                                <Route path="/posts/:slug" element={<BlogPost />} />
                                {/* Fallback for undefined routes */}
                                <Route path="*" element={<NotFoundPage />} />
                            </Routes>
                        </Suspense>
                    </Box>
                    <Footer />
                </Box>
            </Box>
        </>
    );
};

const App = () => {
    return (
        <ThemeProvider>
            <Router>
                <AppContent />
            </Router>
        </ThemeProvider>
    );
};

export default App;
