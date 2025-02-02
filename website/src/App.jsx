import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Header from './components/Header';
import Footer from './components/Footer';
import HomePage from './components/HomePage';
import AboutMe from './components/AboutMe';
import Box from '@mui/material/Box';
import PlaceholderPage from './components/PlaceHolder';
import ProjectsPage from './components/ProjectsPage';

const App = () => {
    return (
        <Router>
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
                    <Header />
                    <Box
                        component="main"
                        sx={{
                            flex: 1, // Makes the main content stretch between the header and footer
                        }}
                    >
                        <Routes>
                            <Route path="/" element={<HomePage />} />
                            <Route path="/about" element={<AboutMe />} />
                            <Route path="/posts" element={<PlaceholderPage title="Posts" />} />
                            <Route path="/projects" element={<ProjectsPage />} />
                            <Route path="/posts/:postId" element={<PlaceholderPage title="" />} />
                            {/* Fallback for undefined routes */}
                            <Route path="*" element={<PlaceholderPage title="404 Not Found" />} />
                        </Routes>
                    </Box>
                    <Footer />
                </Box>
            </Box>
        </Router>
    );
};

export default App;
