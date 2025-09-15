import React from 'react';
import Box from '@mui/material/Box';
import SEO from './SEO';

const HomePage = () => {
    return (
        <>
            <SEO 
                title="Nickolas Goodis - Software Engineer & Data Scientist"
                description="Software engineer and data scientist from University of Miami. Specializing in React, TypeScript, Python, machine learning, and blockchain development."
                keywords="software engineer, data scientist, React, TypeScript, Python, machine learning, blockchain, University of Miami, portfolio"
                url="https://nickogoodis.com"
                type="website"
            />
            <Box
            sx={{
                position: 'relative', // Relative positioning for layout
                minHeight: 'calc(100vh - 64px)', // Adjust for Header and Footer height
                width: '100%',
                overflow: 'hidden', // Prevent content overflow
                display: 'flex',
                flexDirection: 'row',
                alignItems: 'center',
                justifyContent: 'space-between',
                textAlign: 'left',
                color: 'white',
                padding: '50px 20px', // Adjust for spacing
                zIndex: 1, // Ensure content stays above the global background
            }}
        >
            {/* Text Content */}
            <Box
                sx={{
                    flex: 2, // Allocate more space for text
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'center',
                }}
            >
                <h1
                    style={{
                        fontSize: 'clamp(2rem, 5vw, 4rem)', // Dynamically adjusts between 2rem and 4rem
                        margin: '0',
                    }}
                >
                    Nickolas Goodis
                </h1>
                <h2
                    style={{
                        fontSize: 'clamp(1.5rem, 4vw, 2.5rem)', // Dynamically adjusts between 1.5rem and 2.5rem
                        margin: '10px 0',
                    }}
                >
                    Junior at University of Miami studying CS, Math, and AI
                </h2>
                <h4
                    style={{
                        fontSize: 'clamp(1rem, 3vw, 1.5rem)', // Dynamically adjusts between 1rem and 1.5rem
                        lineHeight: '1.8',
                        margin: '10px 0',
                    }}
                >
                    I am a software engineer and data scientist with a passion for exploring
                    different technologies and creating innovative solutions to complex
                    problems.
                </h4>
            </Box>

            {/* Circular Image */}
            <Box
                sx={{
                    flex: 1, // Allocate space for the image
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                }}
            >
                <img
                    src="/me.jpg"
                    alt="Nickolas Goodis"
                    style={{
                        width: '60%', // Dynamically adjust the size relative to its container
                        height: 'auto', // Maintain aspect ratio
                        borderRadius: '50%', // Make the image circular
                        border: '4px solid white', // Optional border styling
                    }}
                />
            </Box>
        </Box>
        </>
    );
};

export default HomePage;
