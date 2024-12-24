import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Header from './components/Header';
import AboutMe from './components/AboutMe';
import Footer from './components/Footer';
import HomePage from './components/HomePage';
import Box from '@mui/material/Box';

const App = () => {
    return (
        <Router>
            {/* Background Wrapper */}
            <Box
                sx={{
                    backgroundColor: 'rgba(0, 80, 48, 0.3)', // Background overlay color
                    position: 'relative',
                    minHeight: '100vh', // Full viewport height
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
                        zIndex: -1, // Push the background behind all other content
                    }}
                />

                {/* App Content */}
                <Header />
                <Routes>
                    <Route path="/" element={<HomePage />} />
                    <Route path="/about" element={<AboutMe />} />
                </Routes>
                <Footer />
            </Box>
        </Router>
    );
};

export default App;
