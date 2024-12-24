import React from 'react';
import Box from '@mui/material/Box';

const HomePage = () => {
    return (
        <Box
            sx={{
                backgroundColor: 'rgba(0, 80, 48, 0.3)',
                position: 'relative', // Relative positioning for layout
                minHeight: 'calc(100vh - 64px)', // Adjust for Header and Footer height (if header/footer aren't fixed)
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
                    backgroundImage: 'url(campus1.jpg)', // Path to your background image
                    backgroundSize: 'cover',
                    backgroundRepeat: 'no-repeat',
                    backgroundPosition: 'center',
                    zIndex: -1, // Push it behind other elements
                }}
            />

            {/* Main Content */}
            <Box
                sx={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'left',
                    justifyContent: 'center',
                    textAlign: 'left',
                    color: 'white',
                    padding: '100px 20px', // Adjust for spacing
                    zIndex: 1, // Ensure content stays above the background
                }}
            >
                <h1>Nickolas Goodis</h1>
                <h2>Junior at University of Miami studying CS, Math, and AI</h2>
                <h4>I am a software engineer and data scientist with a passion for creating innovative solutions to complex problems.</h4>
            </Box>
        </Box>
    );
};

export default HomePage;
