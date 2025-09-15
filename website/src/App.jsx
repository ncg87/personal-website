import React from 'react';
import { BrowserRouter as Router, Route, Routes, useLocation } from 'react-router-dom';
import ModernHeader from './components/ModernHeader';
import Footer from './components/Footer';
import TerminalHomePage from './components/TerminalHomePage';
import ModernAboutPage from './components/ModernAboutPage';
import Resume from './components/Resume';
import ContactForm from './components/ContactForm';
import Box from '@mui/material/Box';
import PlaceholderPage from './components/PlaceHolder';
import ModernProjectsPage from './components/ModernProjectsPage';
import ProjectCaseStudy from './components/ProjectCaseStudy';
import SkipLink from './components/ui/SkipLink';

const AppContent = () => {
    const location = useLocation();
    
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
                        <Routes location={location} key={location.pathname}>
                            <Route path="/" element={<TerminalHomePage key="homepage" />} />
                            <Route path="/about" element={<ModernAboutPage />} />
                            <Route path="/resume" element={<Resume />} />
                            <Route path="/contact" element={<ContactForm />} />
                            <Route path="/posts" element={<PlaceholderPage title="Posts" />} />
                            <Route path="/projects" element={<ModernProjectsPage />} />
                            <Route path="/projects/:slug" element={<ProjectCaseStudy />} />
                            <Route path="/posts/:postId" element={<PlaceholderPage title="" />} />
                            {/* Fallback for undefined routes */}
                            <Route path="*" element={<PlaceholderPage title="404 Not Found" />} />
                        </Routes>
                    </Box>
                    <Footer />
                </Box>
            </Box>
        </>
    );
};

const App = () => {
    return (
        <Router>
            <AppContent />
        </Router>
    );
};

export default App;
