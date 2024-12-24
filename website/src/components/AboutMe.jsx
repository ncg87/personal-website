import React from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';

const AboutMe = () => {
    return (
        <Box
            id="about"
            sx={{
                position: 'relative', // Matches the positioning from HomePage
                minHeight: 'calc(100vh - 64px)', // Adjusts for Header and Footer height
                width: '100%',
                overflow: 'hidden', // Prevents content overflow
                display: 'flex',
                flexDirection: { xs: 'column', md: 'row' }, // Stacks content on small screens
                alignItems: 'center',
                justifyContent: 'space-between',
                textAlign: 'left',
                padding: { xs: '20px', md: '50px' }, // Responsive padding
                color: 'white',
                backgroundColor: 'rgba(30, 30, 30, 0.9)', // Semi-transparent dark background
            }}
        >
            {/* Text Section */}
            <Box
                sx={{
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'center',
                    maxWidth: { xs: '100%', md: '60%' }, // Responsive width
                    padding: { xs: '10px', md: '20px' }, // Responsive padding
                }}
            >
                <Typography
                    variant="h4"
                    component="h2"
                    sx={{
                        fontSize: { xs: '1.8rem', md: 'clamp(2rem, 5vw, 3rem)' }, // Responsive font size
                        marginBottom: '20px',
                        color: '#1abc9c', // Accent color for heading
                    }}
                >
                    About Me
                </Typography>
                <Typography
                    variant="body1"
                    sx={{
                        fontSize: { xs: '1rem', md: 'clamp(1rem, 2.5vw, 1.5rem)' }, // Responsive font size
                        lineHeight: 1.8,
                        textAlign: 'justify',
                        marginBottom: '20px',
                    }}
                >
                    I'm a passionate and driven student at the University of Miami,
                    pursuing a triple major in Computer Science, Mathematics, and
                    Data Science & AI. With a strong academic background (GPA: 3.9)
                    and hands-on experience in machine learning and software
                    development, I'm dedicated to pushing the boundaries of
                    technology and innovation. My research work and projects span
                    from advanced machine learning applications to quantitative
                    analysis and mobile app development.
                </Typography>
            </Box>
        </Box>
    );
};

export default AboutMe;
