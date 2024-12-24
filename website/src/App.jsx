import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Header from './components/Header';
import Footer from './components/Footer';
import HomePage from './components/HomePage';
import AboutMe from './components/AboutMe';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';

const PlaceholderPage = ({ title }) => (
    <Box
        sx={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            minHeight: 'calc(100vh - 64px)', // Ensure it fills the screen minus the header and footer height
            padding: '20px',
            color: 'white',
            textAlign: 'center',
            position: 'relative',
            backgroundColor: 'rgba(0, 0, 0, 0.5)', // Optional overlay
        }}
    >
        {/* Placeholder Content */}
        <Typography
            variant="h1"
            sx={{
                fontSize: 'clamp(2rem, 5vw, 4rem)', // Responsive font size
                fontWeight: 'bold',
                zIndex: 1, // Ensures the text appears above the background
            }}
        >
            {title} Page Coming Soon
        </Typography>
    </Box>
);




const App = () => {
    return (
        <Router>
            {/* App Container */}
            <Box
                sx={{
                    backgroundColor: 'rgba(0, 80, 48, 0.3)', // Background overlay color
                    position: 'relative',
                    minHeight: '100%', // Ensures the app spans the entire viewport
                    width: '100%',
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
                            <Route path="/contact" element={<PlaceholderPage title="Contact" />} />
                            <Route path="/projects" element={<PlaceholderPage title="Projects" />} />
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
